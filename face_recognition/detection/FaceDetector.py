import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
import onnx
import onnxruntime as ort

from face_recognition.detection.config import cfg_mnet, cfg_re50
from face_recognition.detection.box.prior_box import PriorBox
from face_recognition.detection.nms.py_cpu_nms import py_cpu_nms
from face_recognition.detection.box.box_utils import decode, decode_landm


class FaceDetector:
    def __init__(self, weight_path: str, input_size: Tuple[int, int], network='mobile0.25',
                 confidence_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):
        """
        Args:
            weight_path:
            input_size: (height, width)
            network: default mobile0.25
            confidence_threshold:
            top_k:
            nms_threshold:
            keep_top_k:
            vis_thres:
        """

        self.input_size = input_size
        # load model and configure of model
        self.network = network
        model = onnx.load(weight_path)
        providers = ['CPUExecutionProvider']
        providers_options =[{
            'device_id': '0',
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cuda_mem_limit': str(1 >> 15),
            'cudnn_conv_algo_search': 'HEURISTIC'}]
        self.ort_session = ort.InferenceSession(weight_path, providers=providers)
        self.ort_input = self.ort_session.get_inputs()[0].name
        self.ort_output = self.ort_session.get_outputs()[0].name

        self.cfg = None
        if self.network == "mobile0.25":
            setattr(self, 'cfg', cfg_mnet)
        elif self.network == "resnet50":
            setattr(self, 'cfg', cfg_re50)
        else:
            raise (Exception("Invalid NetWork"))
        self.cfg['pretrain'] = False

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
        img = np.transpose(img, [2, 0, 1])
        img = np.array(img, dtype=img.dtype, order='C')
        batch_a = np.expand_dims(img, axis=0)

        return batch_a

    def detect(self, image_raw):
        """
            Detect face from single image
            :param image_raw: ndarray of image
            :return:
        """

        batch = self.preprocess(image_raw)

        # forward pass
        output = self.ort_session.run(None, {self.ort_input:batch})
        loc, conf, landmarks = output[0], output[1], output[2]
        # start = time.time()
        loc, conf, landmarks = loc.squeeze(0), conf.squeeze(0), landmarks.squeeze(0)

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


if __name__ == '__main__':
    from face_recognition.utils import download_weights
    from face_recognition.utils import Cfg
    config = Cfg.load_config()
    detection_model_path = download_weights(config['weights']['face_detections']['FaceDetector_480_onnx'])
    face_detector = FaceDetector(weight_path=detection_model_path, input_size=(480, 640))
    input_fake = np.random.randint(0, 255, size=(480, 640, 3))

    dets = face_detector.detect(input_fake)
    print(dets.shape)
