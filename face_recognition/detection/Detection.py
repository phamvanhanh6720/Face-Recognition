import math

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms

from face_recognition.utils import download_weights
from face_recognition.detection.utils import BBox
from face_recognition.utils import Cfg

preprocess = transforms.Compose([
    transforms.Resize([56, 56]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class Detection:
    def __init__(self, cfg: dict):
        caffe_model = download_weights(cfg['weights']['face_detections']['caffe_weight'])
        deploy = download_weights(cfg['weights']['face_detections']['file_config'])
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffe_model)
        self.detector_confidence = 0.5

        landmark_weight = download_weights(cfg['weights']['face_detections']['landmark_weight'])
        self.ort_session_landmark = ort.InferenceSession(landmark_weight)
        self.out_size_lm = 56

    def infer(self, ori_image: np.ndarray):
        img = ori_image.copy()
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])

        if out[max_conf_index, 2] >= self.detector_confidence:
            x1, y1, x2, y2 = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                       out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
            bbox = [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)]

            # normalize box to square shape
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h]) * 1.1)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))

            new_bbox = BBox(new_bbox)
            cropped = ori_image[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)

            cropped_face = cv2.resize(cropped, (self.out_size_lm, self.out_size_lm))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face = Image.fromarray(cropped_face)
            test_face = preprocess(cropped_face)
            test_face.unsqueeze_(0)

            ort_inputs = {self.ort_session_landmark.get_inputs()[0].name: to_numpy(test_face)}
            ort_outs = self.ort_session_landmark.run(None, ort_inputs)

            landmark = ort_outs[0]
            landmark = landmark.reshape(-1, 2)
            landmark = new_bbox.reproject_landmark(landmark)

            lefteye_x = 0
            lefteye_y = 0
            for i in range(36, 42):
                lefteye_x += landmark[i][0]
                lefteye_y += landmark[i][1]
            lefteye_x = lefteye_x / 6
            lefteye_y = lefteye_y / 6
            lefteye = [lefteye_x, lefteye_y]

            righteye_x = 0
            righteye_y = 0
            for i in range(42, 48):
                righteye_x += landmark[i][0]
                righteye_y += landmark[i][1]
            righteye_x = righteye_x / 6
            righteye_y = righteye_y / 6
            righteye = [righteye_x, righteye_y]

            nose = landmark[33]
            leftmouth = landmark[48]
            rightmouth = landmark[54]
            facial5points = [righteye, lefteye, nose, rightmouth, leftmouth]

            return bbox, facial5points

        return None, None


def draw_bbox(img, bbox, landmarks):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
    for x, y in landmarks:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img


if __name__ == '__main__':
    url = 'http://192.168.1.9:4747/video'
    cap = cv2.VideoCapture(url)
    config = Cfg.load_config()

    face_detector = Detection(config)
    while True:
        ret, orig_img = cap.read()
        if ret is False:
            break
        height, width, _ = orig_img.shape

        bounding_box, facial5landmarks = face_detector.infer(orig_img.copy())
        if bounding_box is not None:
            draw_bbox(orig_img, bounding_box, facial5landmarks)

        cv2.imshow('test', orig_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
