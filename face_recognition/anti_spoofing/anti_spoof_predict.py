import onnxruntime as ort
import numpy as np
import torch.nn.functional as F
from scipy.special import softmax

from face_recognition.anti_spoofing.data_io import transform as trans
from memory_profiler import profile

@profile()
class AntiSpoofPredict:
    def __init__(self, model_path):
        providers = ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(model_path, providers=providers)
        self.ort_input = self.ort_session.get_inputs()[0].name
        self.ort_output = self.ort_session.get_outputs()[0].name

    def predict(self, img):
        test_transform = trans.Compose([
            trans.ToTensor(), ])
        img = test_transform(img)
        img = img.unsqueeze(0)
        img = img.numpy()

        result = self.ort_session.run([self.ort_output], {self.ort_input: img})[0]
        print(result.shape)
        result = softmax(result)

        return result
