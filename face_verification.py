from models.model_irse import IR_50
import torch
from detect.detect import FaceDetector
import cv2
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

from align.align_trans import warp_and_crop_face, get_reference_facial_points


class FaceVerification:
    def __init__(self, cpu=True, crop_size=112, weights_path="./weights/backbone_ir50_asia.pth"):
        self.cpu = cpu
        self.crop_size = crop_size
        self.weights_path = weights_path

        torch.set_grad_enabled(False)
        self.device = torch.device('cpu' if self.cpu else 'cuda:0')

        # Feature Extraction Model
        self.arcface_r50_asian = IR_50([self.crop_size, self.crop_size])
        self.arcface_r50_asian.load_state_dict(torch.load(weights_path, map_location='cpu' if self.cpu else 'cuda'))
        self.arcface_r50_asian.eval()
        self.arcface_r50_asian.to(self.device)

        # Align
        self.scale = self.crop_size / 112.
        self.reference = get_reference_facial_points(default_square=True) * self.scale

        # Facial Detection Model
        self.face_detector = FaceDetector()

    def preprocess(self, input_image):
        img = np.float32(input_image / 255.)
        img = (img - 0.5) / 0.5

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        if self.cpu:
            model_input = torch.FloatTensor(img)
        else:
            model_input = torch.cuda.FloatTensor(img)

        return model_input

    def l2_normalize(self, embedding):
        return embedding / np.sqrt(np.sum(embedding**2))

    def calculate_similarity(self, image1, image2):
        bboxes1 = self.face_detector.detect(image1)
        bboxes2 = self.face_detector.detect(image2)

        facial5points1 = [[bboxes1[0][2 * i - 1], bboxes1[0][2 * i]] for i in range(3, 8)]
        warped_face1 = warp_and_crop_face(
            cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), facial5points1, self.reference, (self.crop_size, self.crop_size))

        facial5points2 = [[bboxes2[0][2 * i - 1], bboxes2[0][2 * i]] for i in range(3, 8)]
        warped_face2 = warp_and_crop_face(
            cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), facial5points2, self.reference, (self.crop_size, self.crop_size))

        input1 = self.preprocess(warped_face1)
        input2 = self.preprocess(warped_face2)

        embedding1 = self.arcface_r50_asian(input1)
        embedding2 = self.arcface_r50_asian(input2)

        embedding1 = embedding1.detach().numpy()
        print(embedding1.shape)
        embedding2 = embedding2.detach().numpy()
        cosin = cosine_similarity(self.l2_normalize(embedding1), self.l2_normalize(embedding2))
        distance = euclidean_distances(embedding1, embedding2)

        return cosin[0, 0], distance[0, 0], warped_face1, warped_face2


if __name__ == '__main__':
    # extraction model
    device = torch.device('cpu')
    arcface_r50_asian = IR_50([112, 112])
    arcface_r50_asian.load_state_dict(torch.load(
        './weights/backbone_ir50_asia.pth',
        map_location='cpu'))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device)

    crop_size = 112
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    # face detector
    face_detector = FaceDetector()

    image_1 = cv2.imread('./4/id.png')
    image_2 = cv2.imread('./4/selfie.png')
    bboxes_1 = face_detector.detect(image_1)
    bboxes_2 = face_detector.detect(image_2)

    facial5points_1 = [[bboxes_1[0][2*i - 1], bboxes_1[0][2*i]] for i in range(3, 8)]
    warped_face_1 = warp_and_crop_face(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB), facial5points_1, reference, crop_size=(crop_size, crop_size))

    facial5points_2 = [[bboxes_2[0][2*i - 1], bboxes_2[0][2*i]] for i in range(3, 8)]
    warped_face_2 = warp_and_crop_face(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB), facial5points_2, reference, crop_size=(crop_size, crop_size))

    """
    input_1 = preprocess(warped_face_1)
    input_2 = preprocess(warped_face_2)
    """
    embedding_1 = arcface_r50_asian(warped_face_1)
    embedding_2 = arcface_r50_asian(warped_face_2)

    cosin = cosine_similarity(embedding_1, embedding_2)
    print("cosin:", cosin)
    distance = euclidean_distances(embedding_1, embedding_2)
    print('distance:', distance)
    cv2.imshow('cropped_1', cv2.cvtColor(warped_face_1, cv2.COLOR_BGR2RGB))
    cv2.imshow('cropped_2', cv2.cvtColor(warped_face_2, cv2.COLOR_BGR2RGB))

    cv2.waitKey(0)


