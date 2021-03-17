from face_recognition import main


if __name__ == '__main__':
    main(tensorrt=True, cam_device=0, input_size=(480, 640))