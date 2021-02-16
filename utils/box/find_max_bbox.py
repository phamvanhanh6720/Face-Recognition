import numpy as np


def find_max_bbox(dets, area_threshold=None):
    """
    Calculate area of all bounding boxes and return maximum bounding box
    Args:
        dets: 2D array contain all bounding boxes, Shape: (n, 15) - (x1, y1, x2, y2, score, 5 landmarks)
        area_threshold:

    Returns:
        maximum bounding box
    """
    x1 = dets[:, 0:1]
    y1 = dets[:, 1:2]
    x2 = dets[:, 2:3]
    y2 = dets[:, 3:4]
    ares = (x2 - x1) * (y2 - y1)
    max_idx = np.argmax(ares, axis=0)[0]
    max_are = ares[max_idx][0]
    if area_threshold != None:
        if max_are < area_threshold:
            return None

    max_bbox = dets[max_idx, :]
    max_bbox = list(map(int, max_bbox))

    return max_bbox
