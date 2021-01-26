import numpy as np
import cv2
import time
from queue import Queue

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
