import cv2
from utils.capture import VideoCaptureThreading
import time

if __name__=='__main__':
    url_http = 'http://admin:dslabneu8@192.168.0.103:80/ISAPI/streaming/channels/102/httppreview'
    cap = VideoCaptureThreading(url_http)
    cap.start()

    t0 = time.time()
    i = 0
    while True:
        _, frame = cap.read()
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    n_frames = i
    print('Frames per second: {:.2f}, with_threading={}'.format(n_frames / (time.time() - t0), True))
    print("Num_frames: {}".format(n_frames))
    print("total time: {:.2f}".format(time.time() - t0))
    cap.stop()
    cv2.destroyAllWindows()