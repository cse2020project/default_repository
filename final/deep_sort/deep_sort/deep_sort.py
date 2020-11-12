import numpy as np
import torch
import math

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

from . import coor

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img, dict, frame):

        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # coordinate - 방향 전달
        coors = []
        # 각도전달
        angle = []
        # 아이디 전달
        identities = []

        # output bbox identities
        outputs = []
        bbox_size=[]
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            x_center = (x2 + x1) / 2
            box_ratio = (x2 - x1) / (y2 - y1)
            box_size = (x2 - x1) * (y2 - y1)
            track_id = track.track_id

            isCloser = True
            if track_id not in dict: isCloser = False
            # 1단계: 위험거리보다 멀리 있는 차량을 필터링
            if isCloser == True and box_size < 1000 or (((x_center > 0 and x_center < 177) or (x_center> 1173 and x_center<1280 )) and box_size<500):
                print(str(track_id)+" 1단계 ")
                isCloser = False
            # 2단계: 가로세로 비율을 비교해 측면차량을 필터링
            if isCloser == True and box_ratio > 1.5:
                print(str(track_id)+" 2단계 ")
                isCloser = False
            '''
            #3-1단계; 멀어지느 차량
            if isCloser == True and track_id in dict and (box_size - dict[track_id][1])/dict[track_id][1] < -0.055:
                print(str(track_id)+" 3단계 ")
                isCloser = False
            #4-1단계; 주차된 차량
                #사람
            if isCloser == True and track_id in dict and abs(box_size-dict[track_id][1])/dict[track_id][1] <=0.055 :
                print(str(track_id)+" 4단계 멈춰있을때 ")
                isCloser = False
            elif isCloser == True and track_id in dict and dict[track_id][2] == False :
                print(str(track_id)+" 4단계의 예외 ")
                outputs.append(np.array([x1, y1, x2, y2, track_id, False], dtype=np.int))  ####outputs도 bbox확인용으로 만든것것
                dict[track_id] = [x_center, box_size, isCloser, box_ratio]
                continue
            if isCloser == True and track_id in dict and abs(dict[track_id][0] - x_center) > 5:
                print(str(track_id)+" 4단계의 움직일때")
                isCloser = False
                # x좌표 변화가 +-5보다 크면 주차된차 (사람 움직일 때)

            '''
            # 3단계: 주차된 차량을 필터링
            if isCloser == True and track_id in dict and abs(dict[track_id][0] - x_center) > 5:
                print(str(track_id)+" 4단계 움직일 때 ")
                isCloser = False
                # x좌표 변화가 +-5보다 크면 주차된차 (사람 움직일 때)
            if isCloser == True and track_id in dict and abs((dict[track_id][1] - box_size) / dict[track_id][1]) < 0.055:
                print(str(track_id)+" 4단계 멈춰있을 때")
                isCloser = False
                # 바운딩박스 크기가 오차포함 일정하면 주차(사람 멈춰있을 때)
            elif track_id in dict and dict[track_id][1] == False:
                print(str(track_id)+" 4단계 예외 ")
                isCloser = False
            # 4단계: 이전 프레임보다 바운딩박스가 작아진, 즉 멀어지는 차량을 필터링
            if isCloser == True and track_id in dict and dict[track_id][1] > box_size:
                print(str(track_id)+" 3단계 ")
                isCloser = False


            # 필터링 끝났는데 남아있으면 다가오는 차량

            if isCloser == True:
                print("\ncar ", track_id, "\'s degree is: ", coor.coor((x2 + x1) / 2) * 57.29578)
            # 5단계: 바운딩박스의 x2좌표가 0-128사이에 있을 경우
            if x2 >= 0 and x2 <= 128:
                continue
            coors.append(coor.coor(x2 + x1) / 2)  # 중점좌표의 라디안값 저장

            bbox_size.append(box_size)
            outputs.append(np.array([x1, y1, x2, y2, track_id, isCloser], dtype=np.int))  ####outputs도 bbox확인용으로 만든것것
            dict[track_id] = [x_center, box_size, isCloser, box_ratio]



        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs, coors, frame, bbox_size

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features