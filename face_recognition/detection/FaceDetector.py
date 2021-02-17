import torch
import numpy as np
import onnxruntime as ort

from face_recognition.detection.config import cfg_mnet, cfg_re50
from face_recognition.detection.box.prior_box import PriorBox
from face_recognition.detection.nms.py_cpu_nms import py_cpu_nms
from face_recognition.detection.box.box_utils import decode, decode_landm


class FaceDetector:
    def __init__(self, network='mobile0.25', cpu=True, onnx_path="./weights/FaceDetector_640.onnx",
                 confidence_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):

        self.network = network
        self.cpu = cpu
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.cfg = None
        if self.network == "mobile0.25":
            setattr(self, 'cfg', cfg_mnet)
        elif self.network == "resnet50":
            setattr(self, 'cfg', cfg_re50)
        else:
            raise (Exception("Invalid NetWork"))
        self.cfg["pretrain"] = False
        self.resize = 1

        self.device = torch.device("cpu" if self.cpu else "cuda")

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
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # forward pass
        loc, conf, landms = self.ort_session.run(None, {self.input_name: img})

        # convert ndarray to tensor pytorch
        loc = torch.from_numpy(loc)
        loc = loc.to(self.device)
        # conf = conf.to(self.device)
        landms = torch.from_numpy(landms)
        landms = landms.to(self.device)

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        scores = np.squeeze(conf, axis=0)[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

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

        return dets
