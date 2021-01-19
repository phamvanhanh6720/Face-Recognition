import torch
from models.model_irse import IR_50

if __name__ == "__main__":
    cpu = False
    weight_path = "./weights/backbone_ir50_asia.pth"

    device = torch.device('cpu' if cpu else 'cuda:0')
    arcface_r50_asian = IR_50([112, 112])
    arcface_r50_asian.load_state_dict(torch.load(weight_path, map_location='cpu' if cpu else 'cuda'))
    arcface_r50_asian.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
    torch_out = arcface_r50_asian(x)

    # Export the model
    torch.onnx.export(arcface_r50_asian, x, "ArcFace_R50.onnx", export_params=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
