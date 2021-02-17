import numpy as np
import time

from face_recognition.anti_spoofing import parse_model_name
from face_recognition.anti_spoofing import CropImage

def detect_spoof(all_models, image_bbox, original_img):

    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name, model in all_models.items():
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": original_img,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = CropImage.crop(**param)
        start = time.time()
        prediction += model.predict(img)
        test_speed += time.time() - start

    idx_spoof = np.argmax(prediction)
    return idx_spoof
