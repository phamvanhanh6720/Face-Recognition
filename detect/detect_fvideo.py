from detect.detect import FaceDetector
import cv2
import time

if __name__ == '__main__':
    faceDetector = FaceDetector()

    camera = cv2.VideoCapture(0)
    count = 0
    start = time.time()

    while True:
        ret, frame = camera.read()
        dets = faceDetector.detect(frame)
        count += 1
        if not ret:
            break
        if count > 10e6:
            count = 0
            start = time.time()
        for b in dets:
            if b[4] < 0.6:
                continue
            text = "{:.2f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        fps = int(count / (time.time() - start))
        cv2.putText(frame, str(fps), (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyWindow()

