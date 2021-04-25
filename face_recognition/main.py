import time
import asyncio
import warnings
from queue import Queue
from typing import List, Optional, Tuple

import cv2
import torch
import onnx
import numpy as np
from torch2trt import torch2trt
from databases import Database
from torchvision import transforms
import onnx_tensorrt.backend as backend
from sklearn.metrics.pairwise import cosine_similarity

from face_recognition.utils import draw_box
from face_recognition.utils import is_chosen, Cfg, download_weights
from face_recognition.utils import track_queue, store_image
from face_recognition.detection import Detection
from face_recognition.align import warp_and_crop_face, get_reference_facial_points
from face_recognition.anti_spoofing import detect_spoof, AntiSpoofPredict
from face_recognition.dao import StudentDAO
from face_recognition.extract_feature import IR_50

# Turn off warnming
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


async def main(cam_device: Optional[int], input_size: Tuple[int, int], area_threshold=12000, cosin_threshold=0.4,
               padding_threshold=5):
    # base configure
    cpu = not torch.cuda.is_available()
    device = torch.device('cpu' if cpu else 'cuda:0')
    config: dict = Cfg.load_config()

    url_db = 'postgresql://{}:{}@{}:{}/{}'.format(config['postgres_db']['user'], config['postgres_db']['password'],
                                                  config['postgres_db']['host'],config['postgres_db']['port'],
                                                  config['postgres_db']['database'])
    database = Database(url_db)
    await database.connect()

    # Load Embeddings
    studentDao = StudentDAO()
    names_list, X = studentDao.load_embeddings()

    # Load Face Detection Model
    print("Load Face Detection Model")
    face_detector = Detection(config)

    # Load reference of alignment
    reference = get_reference_facial_points(default_square=True)

    # Load Face Anti Spoof Models
    print("Load Face Anti Spoof Model")
    anti_spoof_names: List[str] = config['anti_spoof_name']
    model_spoofing = {}
    for model_name in anti_spoof_names:
        path = download_weights(config['weights']['anti_spoof_models'][model_name])
        model_spoofing[model_name] = AntiSpoofPredict(model_path=path)

    # Load Embedding Model
    print("Load Embedding Model")
    device = torch.device('cuda')
    arcface_path = download_weights(config['weights']['embedding_models']['ArcFace_pytorch'])
    arcface_r50_asian = IR_50(input_size=[112, 112])
    arcface_r50_asian.load_state_dict(torch.load(arcface_path, map_location=device))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device=device)

    x = torch.ones((1, 3, 112, 112), device=device)
    arcface_r50_asian_trt = torch2trt(arcface_r50_asian, [x])
    del x
    del arcface_r50_asian
    torch.cuda.empty_cache()

    # Queue for tracking face
    faces_queue = Queue(maxsize=7)
    current_state: Optional[tuple] = None

    # Camera Configure
    # camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)
    url_cam = 'http://192.168.1.9:4747/video'
    camera = cv2.VideoCapture(cam_device)
    count = 0

    while True:
        ret, frame = camera.read()

        im_height, im_width, _ = frame.shape
        if im_height != input_size[0] or im_width != input_size[1]:
            raise Exception('Frame size must be {}'.format(input_size))

        if not ret:
            break

        if count % 6 == 0:
            start = time.time()
            original_img = np.copy(frame)
            bbox, facial5landmarks = face_detector.infer(frame)

            if bbox is not None and is_chosen(bbox, area_threshold=area_threshold):
                coordinate = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, x2, y2
                x1, y1, x2, y2 = coordinate
                if abs(x1) >= padding_threshold and abs(y1) >= padding_threshold and \
                        abs(im_width - abs(x2)) >= padding_threshold and abs(im_height - abs(y2)) >= padding_threshold:
                    # Face Anti Spoofing
                    spoof = detect_spoof(model_spoofing, bbox, original_img)

                    # Get extract_feature
                    warped_face2 = warp_and_crop_face(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), facial5landmarks, reference, (112, 112))

                    input_embedding = preprocess(warped_face2)
                    input_embedding = torch.unsqueeze(input_embedding, 0)
                    input_embedding = input_embedding.to(device)

                    embedding: torch.Tensor = arcface_r50_asian_trt(input_embedding)
                    embedding = embedding.detach().cpu().numpy() if embedding.requires_grad else embedding.cpu().numpy()

                    # Calculate similarity
                    similarity = cosine_similarity(embedding, X)
                    arg_max = np.argmax(similarity, axis=-1)
                    cosin = float(similarity[0, arg_max])

                    if cosin < cosin_threshold:
                        name = "unknown"
                    else:
                        name = names_list[arg_max[0]]

                    # Draw box
                    draw_box(frame, coordinate, cosin, name, spoof)

                    faces_queue.put({'label': name, 'spoof': spoof})

            else:
                faces_queue.put({'label': "None", 'spoof': 0})
            result_tracking = track_queue(faces_queue)
            print("Processing 1 frame: {:.4f}".format(time.time() - start))

            if current_state == result_tracking:
                if current_state[2] is not None:
                    print("Dont push to web client: {}___{}".format(current_state[2], current_state[1]))
                else:
                    print("Dont push to web client")
            else:
                current_state = result_tracking
                if current_state[2] is not None:
                    print("Push to web client: {}___{}".format(current_state[2], current_state[1]))
                    resize_img = cv2.resize(original_img, (0, 0), fx=0.5, fy=0.5)
                    student_id = name
                    room_id = str(1109)
                    await store_image(database, room_id=room_id, student_id=student_id, image=resize_img)

            if len(faces_queue.queue) >= 7:
                faces_queue.get()

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count += 1
        if count > 10000:
            count = 0

    # Disconnect database
    await database.disconnect()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    futures = [main(cam_device=None, input_size=(480, 640))]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(futures))
