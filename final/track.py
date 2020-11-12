import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt
import numpy as np
from deep_sort.deep_sort import alert


# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

#import coor

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def draw_boxes(img, bbox, identities=None, isCloser=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = (0, 0, 255) if isCloser[i]==True else (180, 180, 180)
        label = '{}{:d}'.format("", id)
        label += "T" if isCloser[i]==True else "F"

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    flag = 0  # 비디오 저장시 사용할 플래그
    view_img = True
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    idx = -1
    compare_dict = {}

    # create a new figure or activate an exisiting figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')  # 1,1,1그리드

    for path, img, im0s, vid_cap in dataset:
        #plt
        # Plot origin (agent's start point) - 원점=보행자
        ax.plot(0, 0, color='red', marker='o', markersize=20, alpha=0.3)
        # Plot configuration
        ax.set_rticks([])
        ax.set_rmax(1)
        ax.grid(False)
        ax.set_theta_zero_location("S")  # 0도가 어디에 있는지-S=남쪽
        ax.set_theta_direction(-1)  # 시계방향 극좌표

        img = torch.from_numpy(img).to(device)

        # img 프레임 자르기
        # '''input 이미지 프레임 자르기'''
        img = img[:, 100:260, :]

        # 결과 이미지 프레임 자르기
        #결과 프레임 자르기 (bouding box와 object 매칭 시키기 위해!!)
        im0s = im0s[200:520, :, :]

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        idx+=1
        if (idx%10!=0): # 동영상 길이 유지
            if len(outputs) > 0:
                ori_im = draw_boxes(im0s, bbox_xyxy, identities, isCloser) # 이전 정보로 bbox 그리기
            vid_writer.write(im0s)
            continue

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s #우리는 여기로 감 - 파일경로 전까지의 출력문은 datasets.py에서 삭제해야함

            save_path = str(Path(out) / Path(p).name)
            #print(dataset.frame) #프레임 번호
            #s += '%gx%g ' % img.shape[2:]  # print string #영상 사이즈 출력 (예:640x320) - 삭제가능
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

            #만약 차량이 detect된 경우
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results #개수와 클래스 출력(예: 5 cars) -삭제가능
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                   #s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape #결과프레임의 사이즈
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy) #center좌표, w, h
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    bbox_xywh.append(obj)
                    confs.append([conf.item()])


                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs= []

                if len(bbox_xywh)!=0: #뭔가 detect됐다면 deepsort로 보냄
                    outputs,coors,frame = deepsort.update(xywhs, confss , im0, compare_dict,dataset.frame)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4:5]
                    isCloser = outputs[:, -1]
                    print(compare_dict)
                    ori_im = draw_boxes(im0, bbox_xyxy, identities, isCloser)  # bbox 그리기
                    alert.show_direction(ax,coors,frame) # 방향 display하는 함수 호출

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            plt.show(block=False)
            plt.pause(0.01)
            plt.cla()

            # Stream results
            cv2.imshow('frame', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if(flag==0):
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        h = 320
                        flag=1
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    else:
                        vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5m.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)

# python track.py --weights ./best.pt --source ../360cam_sample/0925samples/VID_20200924_204559_00_069.mp4 --img-size 640 --conf-thres 0.2
# python track.py --weights ./d5_300.pt --source ../../sample4.mp4 --img-size 640 --conf-thres 0.2
# python track.py --weights ./d5_300.pt --source ../../360cam_sample/1004samples/step1_front_3.mp4 --img-size 640 --conf-thres 0.2
# python track.py --weights ./d5_300.pt --source ../../360cam_sample/1004samples/step3_back.mp4 --img-size 640 --conf-thres 0.2
# python track.py --weights ./d5_300.pt --source ../../360cam_sample/1004samples/step3_front_2.mp4 --img-size 640 --conf-thres 0.2
# 차량이 다가올 때, 사람과 차량이 같은 방향으로 움직임 step3_back_2
# 차량이 다가올 때, 사람과 차량이 반대 방향으로 움직임(사람이 차쪽으로 다가감)