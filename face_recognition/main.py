import glob
import time
import warnings
from queue import Queue
from typing import List, Optional

import cv2
import torch
import numpy as np
import unidecode
from torchvision import transforms
try:
    from torch2trt import torch2trt
except Exception as e:
    print(e)
    pass
from sklearn.metrics.pairwise import cosine_similarity

from face_recognition.dao import Connector
from face_recognition.utils import draw_box
from face_recognition.utils import find_max_bbox, Cfg, download_weights
from face_recognition.utils import track_queue, check_change
from face_recognition.detection import FaceDetector
from face_recognition.align import warp_and_crop_face, get_reference_facial_points
from face_recognition.anti_spoofing import detect_spoof, AntiSpoofPredict
from face_recognition.extract_feature import IR_50

# Turn off warnming
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main(tensorrt: bool, cam_device: Optional[int], area_threshold=10000, score_threshold=0.6, cosin_threshold=0.4):

    # Load embeddings and labels
    connector = Connector()
    names_list, X = connector.load_embeddings()

    # base configure
    cpu = not torch.cuda.is_available()
    device = torch.device('cpu' if cpu else 'cuda:0')
    config: dict = Cfg.load_config()

    # Load Face Detection Model
    detection_model_path = download_weights(config['weights']['face_detections']['FaceDetector_pytorch'])
    face_detector = FaceDetector(detection_model_path, cpu=cpu, tensorrt=tensorrt, input_size=(480, 640))

    # Load reference of alignment
    reference = get_reference_facial_points(default_square=True)

    # Load Face Anti Spoof Models
    anti_spoof_names: List[str] = config['anti_spoof_name']
    model_spoofing = {}
    for model_name in anti_spoof_names:
        path = download_weights(config['weights']['anti_spoof_models'][model_name])
        model_spoofing[model_name] = AntiSpoofPredict(model_path=path, model_name=model_name)

    # Load Embedding Model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    arcface_path = download_weights(config['weights']['embedding_models']['ArcFace_pytorch'])
    arcface_r50_asian = IR_50(input_size=[112, 112])
    arcface_r50_asian.load_state_dict(torch.load(arcface_path, map_location=device))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device=device)

    if not cpu and tensorrt:
        x = torch.ones((1, 3, 112, 112), device=device)
        arcface_r50_asian = torch2trt(arcface_r50_asian, [x])

    # Queue for tracking face
    faces_queue = Queue(maxsize=7)
    current_state: Optional[tuple] = None

    # Camera Configure
    camera_url = config['camera_url'] if cam_device is None else None
    camera = cv2.VideoCapture(cam_device) if camera_url is None else cv2.VideoCapture(camera_url)
    count = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        if count % 6 == 0:

            original_img = np.copy(frame)

            bounding_boxes = face_detector.detect(frame)
            remove_rows = list(np.where(bounding_boxes[:, 4] < score_threshold)[0]) # score_threshold
            bounding_boxes = np.delete(bounding_boxes, remove_rows, axis=0)

            if bounding_boxes.shape[0] != 0 and find_max_bbox(bounding_boxes, area_threshold=area_threshold) is not None:
                max_bbox = find_max_bbox(bounding_boxes)

                coordinate = [max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3]]   # x1, y1, x2, y2
                landmarks = [[max_bbox[2 * i - 1], max_bbox[2 * i]] for i in range(3, 8)]

                # Face Anti Spoofing
                # image_bbox: x_top_left, y_top_left, width, height
                image_bbox = [int(max_bbox[0]), int(max_bbox[1]),
                              int(max_bbox[2]-max_bbox[0]), int(max_bbox[3]-max_bbox[1])]
                spoof = detect_spoof(model_spoofing, image_bbox, original_img)

                # Get extract_feature
                warped_face2 = warp_and_crop_face(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks, reference, (112, 112))
                input_embedding = preprocess(warped_face2)
                input_embedding = torch.unsqueeze(input_embedding, 0)
                embedding = arcface_r50_asian(input_embedding)
                embedding = embedding.detach().cpu().numpy() if not cpu else embedding.detach().numpy()

                # Calculate similarity
                similarity = cosine_similarity(embedding, X)
                idx = np.argmax(similarity, axis=-1)
                cosin = float(similarity[0, idx])

                if cosin < cosin_threshold:
                    name = "unknown"
                else:
                    name = unidecode.unidecode(names_list[idx[0]])

                # Draw box
                draw_box(frame, coordinate, cosin, name, spoof)

                faces_queue.put({'label': name, 'spoof': spoof})
            else:
                faces_queue.put({'label': "None", 'spoof': 0})

            result_tracking = track_queue(faces_queue)
            current_state = check_change(result_tracking, current_state)

            if len(faces_queue.queue) >= 7:
                faces_queue.get()

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count += 1
        if count > 10000:
            count = 0

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(tensorrt=False, cam_device=0)