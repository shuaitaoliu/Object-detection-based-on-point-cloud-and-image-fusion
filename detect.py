
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



#fusion
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication


cmap = plt.cm.jet

def read_bin(bin_path, intensity=False):
    "读取kitti bin格式文件点云"
    lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points

def read_calib(calib_path):
    "读取kitti数据集标定文件"
    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))
    imu2lidar_m = np.array(list(map(float, raw[6].split()[1:]))).reshape((3, 4))
    imu2lidar_m = np.vstack((imu2lidar_m, np.array([0, 0, 0, 1])))
    return P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m

def vis_pointcloud(points, colorsl=None):
    "渲染显示雷达点云"
    """
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :return: point_in_camera numpy.ndarray `N x 3`
    """
    app = QApplication(sys.argv)
    if colorsl is not None:
        colorsl = colorsl / 255
        colorsl = np.hstack((colorsl, np.ones(shape=(colorsl.shape[0], 1))))
    else:
        colorsl = (1, 1, 1, 1)
    og_widget = gl.GLViewWidget()
    point_size = np.zeros(points.shape[0], dtype=np.float16) + 0.1

    points_item1 = gl.GLScatterPlotItem(pos=points, size=point_size, color=colorsl, pxMode=False)
    og_widget.addItem(points_item1)

    # 作为对比
    points_item2 = gl.GLScatterPlotItem(pos=points, size=point_size, color=(1, 1, 1, 1), pxMode=False)
    points_item2.translate(0, 0, 20)
    og_widget.addItem(points_item2)

#    og_widget.show()
#    cv2.imwrite('../pointcloud.jpg', og_widget)
#    sys.exit(app.exec_())

def image2camera(point_in_image, intrinsic):
    "图像系到相机系反投影"
    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    u = point_in_image[:, 0]
    v = point_in_image[:, 1]
    z = point_in_image[:, 2]
    x = ((u - intrinsic[0, 2]) * z - intrinsic[0, 3]) / intrinsic[0, 0]
    y = ((v - intrinsic[1, 2]) * z - intrinsic[1, 3]) / intrinsic[1, 1]
    point_in_camera = np.vstack((x, y, z))
    return point_in_camera

def lidar2camera(point_in_lidar, extrinsic):
    "雷达系到相机系投影"
    point_in_lidar = np.hstack((point_in_lidar, np.ones(shape=(point_in_lidar.shape[0], 1)))).T
    point_in_camera = np.matmul(extrinsic, point_in_lidar)[:-1, :]  # (X, Y, Z)
    point_in_camera = point_in_camera.T
    return point_in_camera

def camera2image(point_in_camera, intrinsic):
    "相机系到图像系投影"
    point_in_camera = point_in_camera.T
    point_z = point_in_camera[-1]

    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    point_in_camera = np.vstack((point_in_camera, np.ones((1, point_in_camera.shape[1]))))
    point_in_image = (np.matmul(intrinsic, point_in_camera) / point_z)  # 向图像上投影
    point_in_image[-1] = point_z
    point_in_image = point_in_image.T
    return point_in_image

def lidar2image(point_in_lidar, extrinsic, intrinsic):
    "雷达系到图像系投影  获得(u, v, z)"
    point_in_camera = lidar2camera(point_in_lidar, extrinsic)
    point_in_image = camera2image(point_in_camera, intrinsic)
    return point_in_image

def get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w):
    "获取fov内的点云mask, 即能够投影在图像上的点云mask"
    point_in_image = lidar2image(point_in_lidar, extrinsic, intrinsic)
    front_bound = point_in_image[:, -1] > 0
    point_in_image[:, 0] = np.round(point_in_image[:, 0])
    point_in_image[:, 1] = np.round(point_in_image[:, 1])
    u_bound = np.logical_and(point_in_image[:, 0] >= 0, point_in_image[:, 0] < w)
    v_bound = np.logical_and(point_in_image[:, 1] >= 0, point_in_image[:, 1] < h)
    uv_bound = np.logical_and(u_bound, v_bound)
    mask = np.logical_and(front_bound, uv_bound)
    return point_in_image[mask], mask

def get_point_in_image(point_in_lidar, extrinsic, intrinsic, h, w):
    "把雷达点云投影到图像上, 且经过筛选.  用这个就可以了."
    point_in_image, mask = get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w)
    depth_image = np.zeros(shape=(h, w), dtype=np.float32)
    depth_image[point_in_image[:, 1].astype(np.int32), point_in_image[:, 0].astype(np.int32)] = point_in_image[:, 2]
    return point_in_image, depth_image

def depth_colorize(depth):
    "深度图着色渲染"
    """
    :param depth: numpy.ndarray `H x W`
    :return: numpy.ndarray `H x W x C'  RGB
    """
    assert depth.ndim == 2, 'my_depth image shape need to be `H x W`.'
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth

def get_colored_depth(depth):
    "渲染深度图, depth_colorize函数的封装"
    if len(depth.shape) == 3:
        depth = depth.squeeze()
    colored_depth = depth_colorize(depth).astype(np.uint8)
    colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
    return colored_depth

def render_image_with_depth(color_image, depth_image, max_depth=None):
    "根据深度图渲染可见光图像, 在可见光图像上渲染点云"
    """
    :param color_image:  numpy.ndarray `H x W x C`
    :param depth_image:  numpy.ndarray `H x W`
    :param max_depth:  int 控制渲染效果
    :return:
    """
    depth_image = depth_image.copy()
    if max_depth is not None:
        depth_image = np.minimum(depth_image, max_depth)
    color_image = color_image.copy()
    colored_depth = get_colored_depth(depth_image)
    idx = depth_image != 0
    color_image[idx] = colored_depth[idx]
    return color_image



#----------------------------------------------------------------------------------------------------------------
#detect
@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
#fusion--------------------------------------------------------------------------------------------------------
    image_path = './data_example/3d_detection/image_2/000007.png'
    bin_path = './data_example/3d_detection/velodyne/000007.bin'
    calib_path = './data_example/3d_detection/calib/000007.txt'
    point_in_lidar = read_bin(bin_path)
    color_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    _, _, P2, _, R0, lidar2camera_matrix, _ = read_calib(calib_path)
    intrinsic = P2  # 内参
    extrinsic = np.matmul(R0, lidar2camera_matrix)  # 雷达到相机外参
    h, w = color_image.shape[:2]  # 图像高和宽

    point_in_image, mask = get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w)
    valid_points = point_in_lidar[mask]

    # 获取颜色
    colorsl = color_image[point_in_image[:, 1].astype(np.int32),
                          point_in_image[:, 0].astype(np.int32)]  # N x 3
    colored_point = np.hstack((valid_points, colorsl))  # N x 6

    # 获取深度图
    sparse_depth_image = np.zeros(shape=(h, w), dtype='float32')
    sparse_depth_image[point_in_image[:, 1].astype(np.int32),
                       point_in_image[:, 0].astype(np.int32)] = point_in_image[:, 2]
    colored_sparse_depth_image = get_colored_depth(sparse_depth_image)
    rendered_color_image = render_image_with_depth(color_image, sparse_depth_image)

    #    cv2.imshow('colored_sparse_depth', colored_sparse_depth_image)
    #    cv2.imshow('rendered_color_image', rendered_color_image.astype(np.uint8))

    #    cv2.imshow('color_image', color_image)
    #    cv2.imshow('sparse_depth', sparse_depth_image)

    cv2.imwrite('./data_example/my_depth/my_depth.jpg', colored_sparse_depth_image)
    cv2.imwrite('./data/images/project.jpg', rendered_color_image.astype(np.uint8))

    vis_pointcloud(points=valid_points, colorsl=colorsl)

#detect------------------------------------------------------------------------------------------------------
    opt = parse_opt()
    main(opt)
