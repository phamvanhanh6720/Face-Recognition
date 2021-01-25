import cv2
from utils.capture import VideoCaptureThreading
import time

if __name__=='__main__':
    camera = cv2.VideoCapture('http://admin:dslabneu8@192.168.0.103:80/ISAPI/streaming/channels/102/httppreview')
    t0 = time.time()
    i = 0
    while True:
        _, frame = camera.read()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
        if time.time() - t0 > 1:
            print(i)
            t0 = time.time()
            i = 0