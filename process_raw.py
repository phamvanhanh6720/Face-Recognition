import cv2
import os
from argparse import ArgumentParser
import numpy as np
from FaceDetector import FaceDetector
from align.align_trans import warp_and_crop_face, get_reference_facial_points
import glob

if __name__=="__main__":

    parser = ArgumentParser(description="Process raw dataset")
    parser.add_argument("--raw_dataset", "-raw_dataset", default="/home/phamvanhanh/PycharmProjects/FaceRecognition/dataset/Face", type=str)
    parser.add_argument("--cpu", '-cpu', default=True, type=bool)
    parser.add_argument("--output_folder", "-output_folder", default="/home/phamvanhanh/PycharmProjects/FaceRecognition/dataset/images")
    parser.add_argument("-crop_size", "--crop_size", default=112, type=int)

    args = parser.parse_args()

    faceDetector = FaceDetector(
        trained_model="/home/phamvanhanh/PycharmProjects/FaceVerification/weights/mobilenet0.25_Final.pth")

    scale = args.crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    imgs = glob.glob(args.raw_dataset + "/*.jpg")
    print(imgs)

    for img_path in imgs:
        name =  (img_path.split("/")[-1]).split(".")[0]

        img = cv2.imread(img_path)

        dets = faceDetector.detect(img)
        original_img = np.copy(img)
        count = 0
        for b in dets:
            if b[4] < 0.6:
                continue

            b = list(map(int, b))
            landmarks = ([[b[2 * i - 1], b[2 * i]] for i in range(3, 8)])

            warped_face1 = warp_and_crop_face(
                original_img, landmarks, reference, (args.crop_size, args.crop_size))
            file_name = os.path.join(args.output_folder, name) + "/" +str(count) + ".jpg"
            print(file_name)
            os.mkdir(os.path.join(args.output_folder, name))
            cv2.imwrite(os.path.join(args.output_folder, name) + "/" +str(count) + ".jpg", warped_face1)

            count +=1

            break