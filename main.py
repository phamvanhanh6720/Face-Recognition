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
from queue import Queue

from datetime import datetime
import warnings
# Turn off warnming
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def preprocess(input_image, cpu=True):
    img = np.float32(input_image / 255.)
    img = (img - 0.5) / 0.5

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

   # Top right
    cv2.line(img, (x2 - r -d , y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def tracking(face_queue: Queue):
    """
    Args:
        face_queue: Contain some elements: {label: str, spoof: int}
        spoof: 0-fake 2d, 1-real, 2-fake 3d
        special case: {label:"None", spoof: None} => Frame doesn't contain face
    Returns:
        True: push to web client
        False: dont push
    """

    elements = list(face_queue.queue)
    # print(elements)
    # key: label, value: list contain 3 values: [num_existence in queue,num_real, num_fake]
    unique_label = dict()
    unique_label["None"] = [0, 0, 0]

    for e in elements:
        l = e['label']
        spoof = e['spoof']

        if l not in unique_label.keys():
            unique_label[l] = [1, 1, 0] if spoof == 1 else [1, 0, 1]

        elif l != "None":
            num_real = unique_label[l][1]
            num_fake = unique_label[l][2]
            num_existence = unique_label[l][0]

            if spoof == 1:
                unique_label[l] = [num_existence+1, num_real+1, num_fake]
            else:
                unique_label[l] = [num_existence+1, num_real, num_fake+1]
        else:
            temp = unique_label["None"][0]
            unique_label["None"] == [temp+1, 0, 0]

    # print("uni", unique_label)
    # Voting
    max_existence = -1
    name = "None"
    for key in unique_label.keys():
        if max_existence < unique_label[key][0]:
            name = key
            max_existence = unique_label[key][0]

    # print(name)
    if name != "None":
        if [unique_label[key][0] for key in unique_label.keys()].count(max_existence) >= 2:
            return 0, ""

        # num_fake >= num_real
        if unique_label[name][2] >= unique_label[name][1]:
            return 0, "{}: fake {:.2f}".format(name, time.time())

        return 1, "{}: real {:.2f}".format(name, time.time())

    else:
        return 0, ""


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

    url_http = 'http://admin:dslabneu8@192.168.0.103:80/ISAPI/Streaming/channels/102/httppreview'

    face_queue = Queue(maxsize=5) # for tracking face
    face_stack = []

    # camera = cv2.VideoCapture(url_http)
    camera = cv2.VideoCapture(0)
    count =0

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        if count % 6 == 0:

            original_img = np.copy(frame)

            start = time.time()
            dets = faceDetector.detect(frame)
            # print("Face detection Time: {:.4f}".format(time.time() - start))

            batch = []  # contain all facial of an image
            coordinates = []
            labels = [] # labels of all facial in an image
            scores = []
            save_imgs = []

            for b in dets:
                if b[4] < 0.6:
                    continue

                score = b[4]  # confidence of a bounding box
                # print(image_bbox, b[:4])
                b = list(map(int, b))
                coordinates.append((b[0], b[1], b[2], b[3])) # x1, y1, x2, y2

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
                    color = (255, 0, 0)  #blue
                else:
                    color = (0, 0, 255)  # red

                cv2.rectangle(
                    frame, (coordinates[max_i][0], coordinates[max_i][1]),
                    (coordinates[max_i][2], coordinates[max_i][3]), color, 2)
                cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
                filename = labels[max_i] + " {:.2f} ".format(scores[max_i]) + str(time.time()) + ".jpg"

                if labels[max_i] != "unknown":
                    cv2.imwrite("./dataset/prediction" + "/" + filename, cv2.cvtColor(save_imgs[max_i], cv2.COLOR_RGB2BGR))

                face_queue.put({'label':labels[max_i], 'spoof': spoof})

                # print("1 frames: {:.4f}".format(time.time()-start1))

            else:

                face_queue.put({'label': "None", 'spoof':0})

            signal, info = tracking(face_queue)
            # print(face_queue.queue)

            if signal == 0 and info == "":
                print("Dont push to client")

            elif info.split(":")[0] == "unknown":
                print("Dont push to client")

            else:
                name = info.split(":")[0]
                spoof = info.split(" ")[1]
                res = {"label": name, "spoof": spoof}
                if face_stack == []:
                    face_stack.append(res)
                    print("Push to web client: " + info)

                elif face_stack[-1] == res:
                    print("Dont push to client " + info)
                else:
                    face_stack.pop()
                    face_stack.append(res)
                    print("Push to web client: " + info)

            if len(face_queue.queue) >= 4:
                face_queue.get()

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count += 1
        if count > 10000:
            count = 0

    camera.release()
    cv2.destroyAllWindows()