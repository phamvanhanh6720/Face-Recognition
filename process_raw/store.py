import os
import glob
import pickle
import codecs
from typing import List, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

from process_raw.dao import Connector
from process_raw.utils import FaceDetector, Cfg
from face_recognition.align import warp_and_crop_face, get_reference_facial_points
from face_recognition.extract_feature import IR_50
from face_recognition.utils import download_weights, find_max_bbox


def store_all(datasets: str):
    """
    Store all id, name, cropped images and embeddings  to database

    Args:
        datasets: file path to specify dataset folders, which contains some directories whose name is id of a person

    Returns: None

    """
    device = torch.device('cpu')
    # load config
    config = Cfg.load_config()
    connector = Connector()

    # load face detection
    detection_model_path = download_weights(config['weights']['face_detections']['FaceDetector_pytorch'])
    face_detector = FaceDetector(detection_model_path)

    # load reference of alignment
    reference = get_reference_facial_points(default_square=True)

    # Load Embedding Model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    arcface_path = download_weights(config['weights']['embedding_models']['ArcFace_pytorch'])
    arcface_r50_asian = IR_50(input_size=[112, 112])
    arcface_r50_asian.load_state_dict(torch.load(arcface_path, map_location=device))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device=device)

    # name of folder in datasets folder is id of person
    folders = os.listdir(datasets)
    for folder in folders:
        img_paths = glob.glob(os.path.join(datasets, folder) + '/*.jpg')
        with open(os.path.join(datasets, folder, 'name.txt'), 'r', encoding='utf-8') as file:
            name = file.read()

        id = int(folder)

        cropped_images = list()
        embeddings = None

        print("Process ID {}".format(id))
        for img_path in img_paths:
            img_a = cv2.imread(img_path)

            bounding_boxes = face_detector.detect(np.copy(img_a))
            remove_rows = list(np.where(bounding_boxes[:, 4] < 0.4))  # score_threshold
            bounding_boxes = np.delete(bounding_boxes, remove_rows, axis=0)

            if bounding_boxes.shape[0] != 0 and find_max_bbox(bounding_boxes) is not None:
                max_bbox = find_max_bbox(bounding_boxes)
                coordinate = [max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3]]  # x1, y1, x2, y2
                landmarks = [[max_bbox[2 * i - 1], max_bbox[2 * i]] for i in range(3, 8)]

                warped_face2 = warp_and_crop_face(
                    cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB), landmarks, reference, (112, 112))
                input_embedding = preprocess(warped_face2)
                input_embedding = torch.unsqueeze(input_embedding, 0)
                embedding = arcface_r50_asian(input_embedding)
                embedding = embedding.detach().numpy()

                if embeddings is None:
                    embeddings = embedding
                    cropped_images.append(warped_face2)
                else:
                    similarity = cosine_similarity(embedding, embeddings)
                    avg_cosine = np.mean(similarity)
                    if avg_cosine >= 0.4:
                        embeddings = np.concatenate([embeddings, embedding], axis=0)
                        cropped_images.append(warped_face2)

        print("Total face: {}".format(len(cropped_images)))
        connector.store_one_person(id=id, name=name, cropped_images=cropped_images, embeddings=embeddings)
        print("-" * 20)


def store_one(folder_path: str):
    """
    Store id, name, cropped_images, embeddings of special person. Folder name is ID of person.

    Args:
        folder_path: file path for specify folder that contains some images and file name.txt

    Returns:

    """
    assert isinstance(int(folder_path.split('/')[-1]), int), "Folder name must be integer and is ID of person"

    device = torch.device('cpu')
    # load config
    config = Cfg.load_config()
    connector = Connector()

    # load face detection
    detection_model_path = download_weights(config['weights']['face_detections']['FaceDetector_pytorch'])
    face_detector = FaceDetector(detection_model_path)

    # load reference of alignment
    reference = get_reference_facial_points(default_square=True)

    # Load Embedding Model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    arcface_path = download_weights(config['weights']['embedding_models']['ArcFace_pytorch'])
    arcface_r50_asian = IR_50(input_size=[112, 112])
    arcface_r50_asian.load_state_dict(torch.load(arcface_path, map_location=device))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device=device)

    images_path = glob.glob(folder_path + '/*.jpg')
    with open(os.path.join(folder_path, 'name.txt'), 'r', encoding='utf-8') as file:
        name = file.read()

    id = int(folder_path.split('/')[-1])
    cropped_images = list()
    embeddings = None
    print("Process ID {}".format(id))


    for img_path in images_path:
        img_a = cv2.imread(img_path)

        bounding_boxes = face_detector.detect(np.copy(img_a))
        remove_rows = list(np.where(bounding_boxes[:, 4] < 0.4))  # score_threshold
        bounding_boxes = np.delete(bounding_boxes, remove_rows, axis=0)

        if bounding_boxes.shape[0] != 0 and find_max_bbox(bounding_boxes) is not None:
            max_bbox = find_max_bbox(bounding_boxes)
            coordinate = [max_bbox[0], max_bbox[1], max_bbox[2], max_bbox[3]]  # x1, y1, x2, y2
            landmarks = [[max_bbox[2 * i - 1], max_bbox[2 * i]] for i in range(3, 8)]

            warped_face2 = warp_and_crop_face(
                cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB), landmarks, reference, (112, 112))
            input_embedding = preprocess(warped_face2)
            input_embedding = torch.unsqueeze(input_embedding, 0)
            embedding = arcface_r50_asian(input_embedding)
            embedding = embedding.detach().numpy()

            if embeddings is None:
                embeddings = embedding
                cropped_images.append(warped_face2)
            else:
                similarity = cosine_similarity(embedding, embeddings)
                avg_cosine = np.mean(similarity)
                if avg_cosine >= 0.4:
                    embeddings = np.concatenate([embeddings, embedding], axis=0)
                    cropped_images.append(warped_face2)

    print("Total face: {}".format(len(cropped_images)))
    connector.store_one_person(id=id, name=name, cropped_images=cropped_images, embeddings=embeddings)


if __name__ == '__main__':
    cropped_images = [np.arange(6)]
    embeddings = np.arange(6)
    store_one('/home/phamvanhanh/PycharmProjects/FaceRecognition/45')







