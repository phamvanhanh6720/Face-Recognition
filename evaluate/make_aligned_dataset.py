import os
import cv2
from detect.detect import FaceDetector
from align.align_trans import warp_and_crop_face, get_reference_facial_points
from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser(description="face alignment and make aligned dataset for evaluate")
    parser.add_argument("-source_root", "--source_root", help="specify your source directory",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/raw_dataset", type=str)
    parser.add_argument("-dest_root", "--dest_root", help="specify your destination directory",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/aligned_dataset", type=str)
    parser.add_argument("-crop_size", "--crop_size", help="specify size of aligned faces", default=112, type=int)
    args = parser.parse_args()

    face_detector = FaceDetector(
        trained_mode="/home/phamvanhanh/PycharmProjects/FaceVerification/weights/mobilenet0.25_Final.pth")

    source_root = args.source_root
    dest_root = args.dest_root
    crop_size = args.crop_size
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))

        count = 0
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = cv2.imread(os.path.join(source_root, subfolder, image_name))

            try:
                bboxes = face_detector.detect(img)
            except Exception:
                print("{} is discarded due to exception !".format(os.path.join(source_root, subfolder, image_name)))
                continue

            if len(bboxes) == 0:
                print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            count += 1
            facial5points = [[bboxes[0][2 * i - 1], bboxes[0][2 * i]] for i in range(3, 8)]
            warped_face = warp_and_crop_face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), facial5points, reference,
                                             crop_size=(crop_size, crop_size))
            cv2.imwrite(os.path.join(dest_root, subfolder, str(count) + ".jpg"),
                        cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR))
            if count == 10:
                break

