import time

import torch
import numpy as np


from face_recognition.detection.load_model import detection_model
from face_recognition.detection.config import cfg_mnet, cfg_re50
from face_recognition.detection.box.prior_box import PriorBox
from face_recognition.detection.nms.py_cpu_nms import py_cpu_nms
from face_recognition.detection.box.box_utils import decode, decode_landm
from face_recognition.detection.retinaface import RetinaFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """
    Old style model is stored with all names of parameters sharing common prefix 'module.
    """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class FaceDetector:
    def __init__(self, trained_model: str, network='mobile0.25', cpu=True,
                 confidence_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):
        self.trained_model = trained_model
        self.network = network
        self.network = network
        self.cpu = cpu
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres

        torch.set_grad_enabled(False)
        self.cfg = None
        if self.network == "mobile0.25":
            setattr(self, 'cfg', cfg_mnet)
        elif self.network == "resnet50":
            setattr(self, 'cfg', cfg_re50)
        else:
            raise (Exception("Invalid NetWork"))

        # build net and load model
        self.net = RetinaFace(self.cfg, phase='test')
        self.net = load_model(self.net, self.trained_model, self.cpu)
        self.net = self.net.eval()

        self.device = torch.device("cpu" if self.cpu else "cuda")
        self.net = self.net.to(self.device)
        self.resize = 1

    def detect(self, image_raw):
        """
        Detect face from single image
        :param image_raw: ndarray of image
        :return:
        """

        img = np.float32(image_raw)
        im_height, im_width, _ = img.shape
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        loc, conf, landmarks = self.net(img)  # forward pass
        loc, conf, landmarks = loc.detach().squeeze(0).numpy(), conf.detach().squeeze(0).numpy()\
            , landmarks.detach().squeeze(0).numpy()

        prior_data = PriorBox(self.cfg, image_size=(im_height, im_width)).forward()

        # decode bounding box
        boxes = decode(loc, prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        scores = conf[:, 1]

        # decode landmarks
        landmarks = decode_landm(landmarks, prior_data, self.cfg['variance'])
        scale1 = np.array([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
        landmarks = landmarks * scale1 / self.resize

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