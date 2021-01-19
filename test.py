import cv2


if __name__=='__main__':
    cap = cv2.VideoCapture('rtsp://admin:dslabneu8@192.168.0.103/1')
    i = 0
    while True:
        i += 1
        ret, img = cap.read()
        if ret:
            cv2.imshow('video output', img)
            # cv2.imwrite('images/{}.jpg'.format(i), img)
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        print("")
    cap.release()
    cv2.destroyAllWindows()