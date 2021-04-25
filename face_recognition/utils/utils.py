import os

import cv2
import gdown


def is_chosen(bbox, area_threshold=None) -> bool:
    """
    Calculate area of bounding boxes and return True if area >= threshold

    Args:
        bbox: (x1, y1, width, heigh)
        area_threshold:

    Returns:
        True/False
    """
    are = bbox[2] * bbox[3]

    if area_threshold is not None:
        if are < area_threshold:
            return False

    return True


def download_weights(url, cache=None, md5=None, quiet=False):

    return os.path.join(gdown.cached_download(url, path=cache, md5=md5, quiet=quiet))


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


def draw_box(img, coordinate, score, name, spoof):
    # print largest face
    cx = coordinate[0]
    cy = coordinate[1] + 12
    text = "{} {:.2f}".format(name, score)

    if spoof == 1:
        color = (255, 0, 0)  # blue
    else:
        color = (0, 0, 255)  # red

    cv2.rectangle(
        img, (coordinate[0], coordinate[1]),
        (coordinate[2], coordinate[3]), color, 2)
    cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))