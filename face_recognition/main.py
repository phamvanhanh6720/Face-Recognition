import os
import glob
import torch
import warnings
from queue import Queue
from typing import List, Optional, Tuple

import torch
import unidecode
import onnxruntime as ort
try:
    from torch2trt import torch2trt
except Exception as e:
    print(e)
    pass
from sklearn.metrics.pairwise import cosine_similarity

from face_recognition.utils.process import *
from face_recognition.utils import find_max_bbox, Cfg, download_weights
from face_recognition.utils import track_queue, check_change
from face_recognition.detection import FaceDetector
from face_recognition.align import warp_and_crop_face, get_reference_facial_points
from face_recognition.anti_spoofing import detect_spoof, AntiSpoofPredict

# Turn off warnming
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main(cam_device=0, tensorrt: bool = True, area_threshold=10000, score_threshold=0.6, cosin_threshold=0.4):

    # Config
    label_file = './dataset/label.txt'
    embeddings_folder = './dataset/embeddings'
    cpu = not torch.cuda.is_available()
    print(cpu)
    device = torch.device('cpu' if cpu else 'cuda:0')


    all_labels = dict()
    try:
        with open(label_file, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                idx = [i for i in range(len(line)) if line[i]==" "]
                name = line[: idx[-1]]
                l = line[idx[-1]:]

                l = int(l.split("\n")[0])
                all_labels[l] = name
    except Exception as e:
        print(e)
    print(all_labels)
    files = glob.glob(embeddings_folder + "/" + "*.npz")

    # embeddings vector of all people in dataset
    X = list()
    y = list()
    for file in files:
        npzfile = np.load(file)
        X.append(npzfile['arr_0'])
        y.append(npzfile['arr_1'])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    config: dict = Cfg.load_config()

    # Load Face Detection Model
    detection_model_path = download_weights(config['weights']['face_detections']['FaceDetector_pytorch'])
    face_detector = FaceDetector(detection_model_path, cpu=cpu, tensorrt=tensorrt)

    # Load reference of Alginment
    reference = get_reference_facial_points(default_square=True)

    # Load Face Anti Spoof Models
    anti_spoof_names: List[str] = config['anti_spoof_name']
    model_spoofing = {}
    for model_name in anti_spoof_names:
        path = download_weights(config['weights']['anti_spoof_models'][model_name])
        model_spoofing[model_name] = AntiSpoofPredict(model_path=path, model_name=model_name)

    # Load Embedding Model
    arcface_onnx_path = download_weights(config['weights']['embedding_models']['ArcFace_R50_onnx'])
    arcface_r50_asian = ort.InferenceSession(arcface_onnx_path)
    input_name = arcface_r50_asian.get_inputs()[0].name

    # Queue for tracking face
    faces_queue = Queue(maxsize=7)
    current_state: Optional[tuple] = None

    # Camera Configure
    camera_url = config['camera_url'] if cam_device is None else None
    print(camera_url)
    camera = cv2.VideoCapture(cam_device) if camera_url is None else cv2.VideoCapture(camera_url)
    count = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        if count % 6 == 0:

            original_img = np.copy(frame)
            bounding_boxes = face_detector.detect(frame)

            remove_rows = list(np.where(bounding_boxes[:, 4] < score_threshold)[0]) # score_thresold
            bounding_boxes = np.delete(bounding_boxes, remove_rows, axis=0)

            if bounding_boxes.shape[0] != 0 and find_max_bbox(bounding_boxes, area_threshold=area_threshold) is not None:
                max_bbox = find_max_bbox(bounding_boxes)

                coordinate = [max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3]]   # x1, y1, x2, y2
                landmarks = [[max_bbox[2 * i - 1], max_bbox[2 * i]] for i in range(3, 8)]

                # Face Anti Spoofing
                # image_bbox: x_top_left, y_top_left, width, height
                image_bbox = [int(max_bbox[0]), int(max_bbox[1]), int(max_bbox[2]-max_bbox[0]), int(max_bbox[3]-max_bbox[1])]
                spoof = detect_spoof(model_spoofing, image_bbox, original_img)

                # Get extract_feature
                warped_face2 = warp_and_crop_face(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks, reference,
                    (112, 112))
                input_embedding = preprocess(warped_face2)
                embedding = arcface_r50_asian.run(None, {input_name: input_embedding})
                embedding = np.array(embedding)
                embedding = np.squeeze(embedding, axis=0)

                # Calculate similarity
                similarity = cosine_similarity(embedding, X)
                idx = np.argmax(similarity, axis=-1)
                cosin = float(similarity[0, idx])

                if cosin < cosin_threshold:
                    name = "unknown"
                else:
                    label = int(y[idx])
                    name = unidecode.unidecode(all_labels[label])

                draw_box(frame, coordinate, cosin, name, spoof)

                faces_queue.put({'label':name, 'spoof': spoof})
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
    main()