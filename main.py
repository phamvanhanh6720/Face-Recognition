import os
import cv2
import torch
from align.align_trans import warp_and_crop_face, get_reference_facial_points
from argparse import ArgumentParser
import glob

import numpy as np
from FaceDetector import FaceDetector
from sklearn.metrics.pairwise import  cosine_similarity
import time
import unidecode
import onnxruntime as ort


def preprocess(input_image, cpu=True):
    img = np.float32(input_image / 255.)
    img = (img - 0.5) / 0.5

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    """    if cpu:
        model_input = torch.FloatTensor(img)
    else:
        model_input = torch.cuda.FloatTensor(img)
    """
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-embeddings_path", "--embeddings_path", default="./dataset/embeddings", type=str)
    parser.add_argument("-label", "--label", default="./dataset/label.txt", type=str)
    parser.add_argument("-cpu", "--cpu", default=True, type=str)
    parser.add_argument("-weight_path", "--weight_path", default="./weights/backbone_ir50_asia.pth", type=str)

    args = parser.parse_args()
    reference = get_reference_facial_points(default_square=True)
    labels = dict()
    try:
        with open(args.label, 'r') as file:
            for line in file.readlines():
                idx = [i for i in range(len(line)) if line[i]==" "]
                name = line[: idx[-1]]
                l = line[idx[-1]:]

                l = int(l.split("\n")[0])
                labels[l] = name
    except Exception as e:
        print(e)

    files = glob.glob(args.embeddings_path + "/" +"*.npz")

    # embeddings vector of all people in dataset
    X = list()
    y = list()
    for file in files:
        npzfile = np.load(file)
        X.append(npzfile['arr_0'])
        y.append(npzfile['arr_1'])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    print(X.shape)
    print(y.shape)
    print(labels)


    faceDetector = FaceDetector()

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    # Feature Extraction Model
    arcface_onnx_path = os.path.join("./weights/ArcFace_R50.onnx")
    arcface_r50_asian = ort.InferenceSession(arcface_onnx_path)
    input_name = arcface_r50_asian.get_inputs()[0].name

    camera = cv2.VideoCapture(0)
    #camera = cv2.VideoCapture('rtsp://admin:dslabneu8@192.168.0.200:554')
    count =0

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        if count % 3 ==0:
            start1 = time.time()
            frame = cv2.resize(frame, (800, 450))
            start = time.time()
            dets = faceDetector.detect(frame)
            print("Face detection Time: {:.4f}".format(time.time() - start))
            original_img = np.copy(frame)

            for b in dets:
                if b[4] < 0.6:
                    continue

                start = time.time()

                score = b[4]

                b = list(map(int, b))
                top_left = (b[0], b[1])
                bottom_right = (b[2], b[3])

                landmarks = [[b[2 * i - 1], b[2 * i]] for i in range(3, 8)]
                warped_face2 = warp_and_crop_face(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks, reference,
                    (112, 112))
                print("Alignment Time: {}".format(time.time()-start))

                start = time.time()
                model_input = preprocess(warped_face2)
                embedding = arcface_r50_asian.run(None, {input_name: model_input})
                embedding = np.array(embedding)
                embedding = np.squeeze(embedding, axis=0)

                # embedding = np.expand_dims(embedding, axis=0)
                # embedding = embedding.detach().numpy()
                print("Embedding time: {}".format(time.time()-start))

                cosins = cosine_similarity(embedding, X)
                idx = np.argmax(cosins)
                cosin = cosins[0, idx]
                if cosin <= 0.35:
                    name = "unknown"
                else:
                    label = int(y[idx])
                    name = unidecode.unidecode(labels[label])

                cx = b[0]
                cy = b[1] + 12
                text = "{} {:.2f}".format(name, cosin)
                # cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                draw_border(frame, top_left, bottom_right, (0, 0, 255), 2, 7, 10)
                cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            print("1 frames: {:.4f}".format(time.time()-start1))

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        count +=1
        if count > 10000:
            count = 0


    camera.release()
    cv2.destroyWindow("frame")