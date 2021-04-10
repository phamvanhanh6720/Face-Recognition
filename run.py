from face_recognition import main


if __name__ == '__main__':
    main(tensorrt=True, cam_device=None, input_size=(480, 640))