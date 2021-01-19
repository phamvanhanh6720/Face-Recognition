import cv2
import onnx

if __name__=='__main__':
    onnx_model = onnx.load("./weights/FaceDetector.onnx")
    onnx.checker.check_model(onnx_model)