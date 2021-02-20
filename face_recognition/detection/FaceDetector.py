import torch
import numpy as np
import time
import onnxruntime as ort
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
    def __init__(self, weight_path, cpu, tensorrt, network='mobile0.25', confidence_threshold=0.02,
                 top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):

        self.network = network
        self.cpu = cpu
        self.device = torch.device('cpu' if cpu else 'cuda:0')
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres

        model = detection_model(weight_path, cpu=cpu, network='mobile0.25')
        model.to(self.device)
        input_size = (1, 3, 480, 640)
        if not cpu and tensorrt:
            x = torch.ones(input_size, device='cpu' if cpu else 'cuda:0')
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
        self.resize = 1
        priorbox = PriorBox(self.cfg, image_size=(480, 640))
        priors = priorbox.forward()
        self.prior_data = priors


    def detect(self, image_raw):
        """
        Detect face from single image
        :param image_raw: ndarray of image
        :return:
        """
        # preprocess input image
        img = np.float32(image_raw)
        try:
            im_height, im_width, _ = img.shape
        except Exception as e:
            print(e)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = scale.to(torch.device('cpu'))
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        batch_a = np.expand_dims(img, axis=0)
        batch = torch.as_tensor(batch_a, device=self.device)

        # forward pass
        loc, conf, landms = self.model(batch)
        start = time.time()
        loc, conf, landms = loc.detach(), conf.detach(), landms.detach()
        if not self.cpu:
            loc = loc.cpu()
            conf = conf.cpu()
            landms = landms.cpu()

        boxes = decode(loc.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.numpy()
        scores = conf.squeeze(0).numpy()[:, 1]
        landms = decode_landm(landms.squeeze(0), self.prior_data, self.cfg['variance'])
        landms = landms.numpy()
        scale1 = np.array([img.shape[2], img.shape[1], img.shape[2], img.shape[1],
                                   img.shape[2], img.shape[1], img.shape[2], img.shape[1],
                                   img.shape[2], img.shape[1]])

        landms = landms * scale1 / self.resize


        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        print("decode time: {}".format(time.time() - start))
        return dets
