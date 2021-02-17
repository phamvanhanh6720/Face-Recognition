import os
import glob
import warnings
from queue import Queue
from argparse import ArgumentParser

import torch
import unidecode
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

from utils.process import *
from utils import tracking
from utils import find_max_bbox
from detection import FaceDetector
from align import warp_and_crop_face, get_reference_facial_points
from anti_spoofing.detect_spoof import detect_spoof
from anti_spoofing.anti_spoof_predict import AntiSpoofPredict

# Turn off warnming
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-embeddings_path", "--embeddings_path", default="./dataset/embeddings", type=str)
    parser.add_argument("-label", "--label", default="./dataset/label.txt", type=str)
    parser.add_argument("-cpu", "--cpu", default=True, type=str)
    parser.add_argument("-weight_path", "--weight_path", default="./weights/backbone_ir50_asia.pth", type=str)

    args = parser.parse_args()
    reference = get_reference_facial_points(default_square=True)
    all_labels = dict()
    try:
        with open(args.label, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                idx = [i for i in range(len(line)) if line[i]==" "]
                name = line[: idx[-1]]
                l = line[idx[-1]:]

                l = int(l.split("\n")[0])
                all_labels[l] = name
    except Exception as e:
        print(e)
    print(all_labels)
    files = glob.glob(args.embeddings_path + "/" + "*.npz")

    # embeddings vector of all people in dataset
    X = list()
    y = list()
    for file in files:
        npzfile = np.load(file)
        X.append(npzfile['arr_0'])
        y.append(npzfile['arr_1'])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # print(X.shape)
    # print(y.shape)

    faceDetector = FaceDetector(onnx_path="./weights/detection_model/FaceDetector_640.onnx")
    model_dir = './weights/anti_spoof_models'
    model_test = {}
    for model_name in os.listdir(model_dir):
        model_test[model_name] = AntiSpoofPredict(os.path.join(model_dir, model_name))
    label = 1

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    # Feature Extraction Model
    arcface_onnx_path = os.path.join("weights/embedding_model/ArcFace_R50.onnx")
    arcface_r50_asian = ort.InferenceSession(arcface_onnx_path)
    input_name = arcface_r50_asian.get_inputs()[0].name

    face_queue = Queue(maxsize=7) # for tracking face
    face_stack = []

    # camera = cv2.VideoCapture(url_http)
    camera = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        if count % 6 == 0:

            original_img = np.copy(frame)
            dets = faceDetector.detect(frame)

            remove_rows = list(np.where(dets[:, 4] < 0.6)[0]) # score_thresold
            dets = np.delete(dets, remove_rows, axis=0)

            if dets.shape[0] != 0:
                max_bbox = find_max_bbox(dets)
                coordinate = [max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3]]   # x1, y1, x2, y2
                landmarks = [[max_bbox[2 * i - 1], max_bbox[2 * i]] for i in range(3, 8)]

                # Face Anti Spoofing
                # image_bbox: x_top_left, y_top_left, width, height
                image_bbox = [int(max_bbox[0]), int(max_bbox[1]), int(max_bbox[2]-max_bbox[0]), int(max_bbox[3]-max_bbox[1])]
                spoof = detect_spoof(model_test, image_bbox, original_img)

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

                if cosin < 0.4:
                    name = "unknown"
                else:
                    label = int(y[idx])
                    name = unidecode.unidecode(all_labels[label])

                draw_box(frame, coordinate, cosin, name, spoof)

                face_queue.put({'label':name, 'spoof': spoof})
            else:
                face_queue.put({'label': "None", 'spoof': 0})

            result_tracking = tracking(face_queue)

            if not result_tracking[0]:
                print("Dont push to client")
                if face_stack != []:
                    face_stack.pop()

            else:
                name = result_tracking[2]
                spoof = result_tracking[1]
                res = [name, spoof]
                if  face_stack == []:
                    face_stack.append(res)
                    print("Push to web client:{} {}".format(name, spoof))

                elif face_stack[-1] == res:
                    print("Dont push to web client:{} {}".format(name, spoof))
                else:
                    face_stack.pop()
                    face_stack.append(res)
                    print("Push to web client:{} {}".format(name, spoof))

            if len(face_queue.queue) >= 7:
                face_queue.get()

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count += 1
        if count > 10000:
            count = 0

    camera.release()
    cv2.destroyAllWindows()