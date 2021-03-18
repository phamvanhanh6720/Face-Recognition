import torch
from pathlib import Path

from face_recognition.extract_feature.model_irse import IR_50
from face_recognition.detection.config import cfg_mnet, cfg_re50
from face_recognition.detection.retinaface import RetinaFace
from face_recognition.anti_spoofing.anti_spoof_predict import AntiSpoofPredict
from face_recognition.detection import load_model


def convert_arc_face_model(weight_path: Path, output_path: Path, crop_size, opset=11):
    """

    Args:
        weight_path:
        output_path:
        crop_size:

    Returns:

    """
    assert crop_size in [112, 224]
    cpu = not torch.cuda.is_available()

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if cpu else 'cuda:0')

    # Feature Extraction Model
    arcface_r50_asian = IR_50([crop_size, crop_size])
    arcface_r50_asian.load_state_dict(torch.load(weight_path, map_location='cpu' if cpu else 'cuda:0'))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device)

    # Export the model
    batch_size = 1
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(batch_size, 3, crop_size, crop_size, device=device)

    torch.onnx.export(arcface_r50_asian, inputs, output_path, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, opset_version=opset)


def convert_retina_face_model(weight_path, output_path, img_size, network="mobile0.25"):
    """
    Args:
        weight_path: absolute path of weigth
        output_path:
        img_size:(width, height)
        network: mobile0.25 or resnet50

    Returns: export onnx model

    """
    h, w = img_size
    torch.set_grad_enabled(False)
    cpu = not torch.cuda.is_available()
    device = torch.device('cpu' if cpu else 'cuda:0')
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, weight_path, cpu)
    net.eval()

    batch_size = 1

    print("==> Exporting model to ONNX format at '{}'".format(output_path))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(batch_size, 3, h, w, device=device)

    torch.onnx.export(net, inputs, output_path, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, opset_version=11)


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
    from face_recognition.utils import download_weights
    from face_recognition.utils import Cfg
    config = Cfg.load_config()
    weigth_path = download_weights(config['weights']['embedding_models']['ArcFace_pytorch'])

    crop_size = 112
    output_path = '../../../Test/TestLoadModel/face_recognition/new_ArcFace_R50.onnx'
    convert_arc_face_model(weigth_path, output_path, crop_size)

