import numpy as np


def preprocess(input_image, cpu=True):
    img = np.float32(input_image / 255.)
    img = (img - 0.5) / 0.5

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img