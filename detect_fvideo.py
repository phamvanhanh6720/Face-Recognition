from FaceDetector import FaceDetector
import cv2
from argparse import ArgumentParser
import os
import numpy as np
import time
from align.align_trans import warp_and_crop_face, get_reference_facial_points
import glob

def crop_image(original_image, top_left, bottom_right):
    cropped_img = original_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

    return cropped_img

if __name__ == '__main__':

    parser = ArgumentParser(description="Extract face from video stream to create dataset")
    parser.add_argument("-name", "--name", default="hanh",type=str)
    parser.add_argument("-default_root", "--default_root",
                        default= "/home/phamvanhanh/PycharmProjects/FaceRecognition/dataset/images", type=str)
    parser.add_argument("-label", "--label", default="/home/phamvanhanh/PycharmProjects/FaceRecognition/dataset/label.txt",
                        help="specify the path of label file", type=str)
    parser.add_argument("-video_path", "--video_path", help="specify your video path to create your dataset", type=str)
    parser.add_argument("-crop_size", "--crop_size", default=112, type=int)

    args = parser.parse_args()
    faceDetector = FaceDetector(trained_model="/home/phamvanhanh/PycharmProjects/FaceVerification/weights/mobilenet0.25_Final.pth")

    # check whether name is used by other people
    if len(glob.glob(args.default_root +"/"+args.name + "/*.jpg")) != 0:
            raise Exception("Please choose other name")
    try:
        os.mkdir(os.path.join(args.default_root, args.name))
    except Exception as e:
        print("Folder was created")
    # Align
    scale = args.crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    camera = cv2.VideoCapture(args.video_path)

    # write name next_id in label
    with open(args.label, 'r') as file:
        length = len(file.readlines())

        print("length:", length)
        next_id = length + 1

    with open(args.label, 'a+') as file:
        try:
            file.write(args.name + " " + str(next_id) + "\n")
        except Exception as e:
            print(e)
            print("Please Specify name")
    images = list()
    landmarks = list()
    count = 0
    start = time.time()
    while True:
        ret, frame = camera.read()
        try:
            dets = faceDetector.detect(frame)
        except Exception as e:
            print(e)
        original_img = np.copy(frame)
        if not ret:
            break

        for b in dets:
            if b[4] < 0.6:
                continue
            text = "{:.2f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            if count % 8 == 0:
                images.append(original_img)
                landmarks.append([[b[2 * i - 1], b[2 * i]] for i in range(3, 8)])

            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        count += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyWindow("DETECT")

    count = 1
    for i in range(len(images)):
        filename = os.path.join(args.default_root, args.name, str(count) + ".jpg")

        facial5points1 = landmarks[i]
        warped_face1 = warp_and_crop_face(
            images[i], facial5points1, reference, (args.crop_size, args.crop_size))
        count += 1
        cv2.imwrite(filename, warped_face1)
