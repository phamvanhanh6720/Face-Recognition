import torch
import numpy as np
import time
from typing import Tuple
from torchvision import transforms
try:
    from torch2trt import torch2trt
except Exception as e:
    print(e)
    pass

from face_recognition.detection.load_model import detection_model
from face_recognition.detection.config import cfg_mnet, cfg_re50
from face_recognition.detection.box.prior_box import PriorBox
from face_recognition.detection.nms.py_cpu_nms import py_cpu_nms
from face_recognition.detection.box.box_utils import decode, decode_landm


class FaceDetector:
    def __init__(self, weight_path: str, cpu: bool, tensorrt: bool, input_size: Tuple[int, int], network='mobile0.25',
                 confidence_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):

        self.input_size = input_size
        self.cpu = cpu
        self.device = torch.device('cpu' if cpu else 'cuda:0')

        # load model and configure of model
        self.network = network
        model = detection_model(weight_path, cpu=cpu, network=self.network)
        model.to(self.device)
        if not cpu and tensorrt:
            temp_size = (1, 3, input_size[0], input_size[1])
            x = torch.ones(temp_size, device='cpu' if cpu else 'cuda:0')
            self.model = torch2trt(model, [x])
        else:
            self.model = model

        self.cfg = None
        if self.network == "mobile0.25":
            setattr(self, 'cfg', cfg_mnet)
        elif self.network == "resnet50":
            setattr(self, 'cfg', cfg_re50)
        else:
            raise (Exception("Invalid NetWork"))
        self.cfg["pretrain"] = False

        # decode configure
        self.resize = 1
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.prior_data = PriorBox(self.cfg, image_size=input_size).forward()
        self.scale1 = np.array([self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]])
        self.scale2 = np.array([self.input_size[1], self.input_size[0], self.input_size[1],self.input_size[0],
                                self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0],
                               self.input_size[1], self.input_size[0]])

    def preprocess(self, image_raw):
        img = np.float32(image_raw)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        batch_a = np.expand_dims(img, axis=0)

        return batch_a

    def detect(self, image_raw):
        """
            Detect face from single image
            :param image_raw: ndarray of image
            :return:
        """

        batch = torch.as_tensor(self.preprocess(image_raw), device=self.device)

        # forward pass
        loc, conf, landmarks = self.model(batch)
        # start = time.time()
        loc, conf, landmarks = loc.detach(), conf.detach(), landmarks.detach()
        if not self.cpu:
            loc = loc.cpu()
            conf = conf.cpu()
            landmarks = landmarks.cpu()

        loc, conf, landmarks = loc.squeeze(0).numpy(), conf.squeeze(0).numpy(), landmarks.squeeze(0).numpy()

        # decode bounding box
        boxes = decode(loc, self.prior_data, self.cfg['variance'])
        boxes = boxes * self.scale1 / self.resize
        scores = conf[:, 1]

        # decode landmarks
        landmarks = decode_landm(landmarks, self.prior_data, self.cfg['variance'])
        landmarks = landmarks * self.scale2 / self.resize

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landmarks = landmarks[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landmarks = landmarks[:self.keep_top_k, :]

        dets = np.concatenate((dets, landmarks), axis=1)

        return dets
