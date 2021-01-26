import os
import cv2
import torch
from align.align_trans import warp_and_crop_face, get_reference_facial_points
from argparse import ArgumentParser
import glob
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.anti_spoof_predict import AntiSpoofPredict, Detection
import numpy as np
from FaceDetector import FaceDetector
from sklearn.metrics.pairwise import  cosine_similarity
import time
import unidecode
import onnxruntime as ort
import copy
from queue import Queue
from utils.utils import preprocess, draw_border, tracking
import threading

# web
from flask import Flask,render_template,jsonify, Response
import random
import base64
from datetime import datetime
import cv2
from datetime import datetime
import warnings
# Turn off warnming
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self.it

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


app = Flask(__name__)
generator = None

@app.route('/')
def main():
    return render_template('index.html', data=[], date=datetime.now().strftime("%d/%m/%Y"))


@app.route('/update_table', methods=['POST', 'GET'])
def updatetable():
    value = next(generator)
    print(value)
    return jsonify(data=value)

@threadsafe_generator
def get_output_to_server():
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
                idx = [i for i in range(len(line)) if line[i] == " "]
                name = line[: idx[-1]]
                l = line[idx[-1]:]

                l = int(l.split("\n")[0])
                all_labels[l] = name
    except Exception as e:
        print(e)
    print(args.label, all_labels)
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

    faceDetector = FaceDetector(onnx_path="./weights/FaceDetector_640.onnx")
    image_cropper = CropImage()
    model_dir = './resources/anti_spoof_models'
    model_test = {}
    for model_name in os.listdir(model_dir):
        model_test[model_name] = AntiSpoofPredict(os.path.join(model_dir, model_name))
    box_detector = Detection()
    label = 1

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    # Feature Extraction Model
    arcface_onnx_path = os.path.join("./weights/ArcFace_R50.onnx")
    arcface_r50_asian = ort.InferenceSession(arcface_onnx_path)
    input_name = arcface_r50_asian.get_inputs()[0].name

    # camera = cv2.VideoCapture(0)
    # camera = cv2.VideoCapture('rtsp://admin:dslabneu8@192.168.0.103:554/')
    camera = cv2.VideoCapture('http://admin:dslabneu8@192.168.0.103:80/ISAPI/streaming/channels/102/httppreview')
    # camera.open(1, apiPreference=cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    camera.set(cv2.CAP_PROP_FPS, 30.0)

    face_queue = Queue(maxsize=5)  # for tracking face
    face_stack = []
    count = 0

    while True:
        bReturn = False
        data = []
        ret, frame = camera.read()

        if not ret:
            yield ""
        else:
            if count % 6 == 0:
                # start1 = time.time()
                frame = cv2.resize(frame, (480, 640))
                original_img = np.copy(frame)
                # frame = cv2.resize(frame, (640, 480))

                start = time.time()
                dets = faceDetector.detect(frame)
                # print("Face detection Time: {:.4f}".format(time.time() - start))

                batch = []  # contain all facial of an image
                coordinates = []
                labels = []  # labels of all facial in an image
                scores = []
                save_imgs = []

                name = ""
                check = False
                now = datetime.now().strftime("%H:%M:%S")
                img = None

                for b in dets:
                    if b[4] < 0.6:
                        continue

                    score = b[4]  # confidence of a bounding box
                    # print(image_bbox, b[:4])
                    b = list(map(int, b))
                    coordinates.append((b[0], b[1], b[2], b[3]))  # x1, y1, x2, y2

                    landmarks = [[b[2 * i - 1], b[2 * i]] for i in range(3, 8)]
                    warped_face2 = warp_and_crop_face(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks, reference,
                        (112, 112))
                    save_imgs.append(warped_face2)

                    # start = time.time()
                    model_input = preprocess(warped_face2)
                    batch.append(model_input)

                if batch != []:

                    start = time.time()
                    image_bbox = box_detector.get_bbox(original_img)
                    prediction = np.zeros((1, 3))
                    test_speed = 0
                    for model_name, model in model_test.items():
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {
                            "org_img": original_img,
                            "bbox": image_bbox,
                            "scale": scale,
                            "out_w": w_input,
                            "out_h": h_input,
                            "crop": True,
                        }
                        if scale is None:
                            param["crop"] = False
                        img = image_cropper.crop(**param)
                        start = time.time()
                        prediction += model.predict(img)
                        test_speed += time.time() - start
                    spoof = np.argmax(prediction)
                    # print("Anti Spoofing Time: {:.4f}".format(time.time() - start))
                    # print('spoof ', spoof)

                    batch = np.concatenate(batch, axis=0)
                    start = time.time()
                    embedding = arcface_r50_asian.run(None, {input_name: batch})

                    embedding = np.array(embedding)
                    embedding = np.squeeze(embedding, axis=0)
                    # print("Embedding time: {}".format(time.time()-start))

                    similarity = cosine_similarity(embedding, X)
                    # print(similarity.shape)
                    idx = np.argmax(similarity, axis=-1)
                    temp = range(0, len(similarity))
                    cosins = similarity[temp, idx]
                    # print(cosins.shape)
                    scores = cosins

                    for i in range(len(cosins)):
                        if cosins[i] <= 0.4:
                            name = "unknown"
                        else:
                            label = int(y[idx[i]])
                            name = unidecode.unidecode(all_labels[label])
                            # print(datetime.fromtimestamp(time.time()),  ": " + name )

                        labels.append(name)

                    # get max box
                    max_i = -1
                    max_area = 0
                    for i in range(len(labels)):
                        area = (coordinates[i][2] - coordinates[i][0]) * (coordinates[i][3] - coordinates[i][1])
                        if area > max_area:
                            max_area = area
                            max_i = i

                    # print largest face
                    cx = coordinates[max_i][0]
                    cy = coordinates[max_i][1] + 12
                    text = "{} {:.2f}".format(labels[max_i], scores[max_i])

                    if spoof == 1:
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)  # red

                    cv2.rectangle(
                        frame, (coordinates[max_i][0], coordinates[max_i][1]),
                        (coordinates[max_i][2], coordinates[max_i][3]), color, 2)
                    cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

                    filename = labels[max_i] + " {:.2f} ".format(scores[max_i]) + str(time.time()) + ".jpg"

                    if labels[max_i] != "unknown":
                        cv2.imwrite("./dataset/prediction" + "/" + filename,
                                    cv2.cvtColor(save_imgs[max_i], cv2.COLOR_RGB2BGR))

                    name = labels[max_i]
                    check = 'Real' if spoof == 1 else 'Fake'
                    now = datetime.now().strftime("%H:%M:%S")
                    face = original_img[coordinates[max_i][1]: coordinates[max_i][3], coordinates[max_i][0]:coordinates[max_i][2]]

                    data.append(name)
                    data.append(now)
                    data.append(face)
                    data.append(check)

                    face_queue.put({'label': labels[max_i], 'spoof': spoof})

                    # print("1 frames: {:.4f}".format(time.time()-start1))

                else:

                    face_queue.put({'label': "None", 'spoof': 0})

                signal, info = tracking(face_queue)
                # print(face_queue.queue)
                if signal == 0 and info == "":
                    # print("Dont push to client")
                    if face_stack != []:
                        face_stack.pop()

                elif info.split(":")[0] == "unknown":
                    # print("Dont push to client")
                    if face_stack != []:
                        face_stack.pop()
                else:
                    name = info.split(":")[0]
                    spoof = info.split(" ")[1]
                    res = {"label": name, "spoof": spoof}
                    if face_stack == []:
                        face_stack.append(res)
                        bReturn = True
                        print("Push to web client: " + info)

                    elif face_stack[-1] == res:
                        pass
                        # print("Dont push to client " + info)
                    else:
                        face_stack.pop()
                        face_stack.append(res)
                        bReturn = True
                        print("Push to web client: " + info)

                if len(face_queue.queue) >= 4:
                    face_queue.get()

                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                     break

            count += 1
            if count > 10000:
                count = 0

            if bReturn and len(data) > 0:
                img = data[2]
                retval, img = cv2.imencode('.jpg', img)
                img = base64.b64encode(img).decode("utf-8")
                data[2] = img
                yield data

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generator = get_output_to_server()
    app.run(debug=True)