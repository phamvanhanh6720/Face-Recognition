from face_recognition import main
import asyncio

if __name__ == '__main__':
    futures = [main(cam_device=None, input_size=(480, 640))]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(futures))