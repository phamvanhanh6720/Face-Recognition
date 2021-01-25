import torch
from argparse import ArgumentParser
from models.model_irse import IR_50
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from src.anti_spoof_predict import AntiSpoofPredict

def convert_arc_face_model(weight_path, output_path, img_size, architecture):
    h, w, c = img_size
    cpu = not torch.cuda.is_available()

    device = torch.device('cpu' if not cpu else 'cuda:0')
    model = architecture([h, w])
    model.load_state_dict(torch.load(weight_path, map_location='cpu' if cpu else 'cuda'))
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, c, h, w, requires_grad=True)

    # Export the model
    torch.onnx.export(model, x, output_path, export_params=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

def convert_retina_face_model(weight_path, output_path, img_size, architecture):
    h, w, c = img_size
    cpu = not torch.cuda.is_available()

    device = torch.device('cpu' if not cpu else 'cuda:0')
    model = architecture(cfg_mnet)
    model.load_state_dict(torch.load(weight_path, map_location='cpu' if cpu else 'cuda'))
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, c, h, w, requires_grad=True)

    # Export the model
    torch.onnx.export(model, x, output_path, export_params=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


def convert_spoof_model(weight_path, output_path, img_size, architecture):
    h, w, c = img_size
    cpu = not torch.cuda.is_available()

    device = torch.device('cpu' if not cpu else 'cuda:0')
    model = AntiSpoofPredict(device_id=0)
    model.load_state_dict(model)
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, c, h, w, requires_grad=True)

    # Export the model
    torch.onnx.export(model, x, output_path, export_params=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


if __name__ == "__main__":
    '''
    parser = ArgumentParser()
    parser.add_argument("-weight_path", "--weight_path", default="./dataset/embeddings", type=str)
    parser.add_argument("-label", "--label", default="./dataset/label.txt", type=str)
    parser.add_argument("-cpu", "--cpu", default=True, type=str)
    parser.add_argument("-weight_path", "--weight_path", default="./weights/backbone_ir50_asia.pth", type=str)
    '''

    weight_path = "./weights/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    output_path = "./weights/MiniFASNetV2.onnx"
    img_size = (640, 480, 3)
    architecture = AntiSpoofPredict
    convert_spoof_model(weight_path, output_path, img_size, architecture)
