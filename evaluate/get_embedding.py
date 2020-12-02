from models.model_irse import IR_50
import torch
import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

def preprocess(input_image, cpu):
    img = np.float32(input_image / 255.)
    img = (img - 0.5) / 0.5

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    if cpu:
        model_input = torch.FloatTensor(img)
    else:
        model_input = torch.cuda.FloatTensor(img)

    return model_input

def l2_normalize(embedding):
    return embedding / np.sqrt(np.sum(embedding ** 2))


if __name__ =='__main__':
    parser = ArgumentParser(description="get embedding")
    parser.add_argument('-dataset_folder', '--dataset_folder',
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/aligned_dataset", type=str)
    parser.add_argument('-embeddings_folder', '--embeddings_folder',
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/embeddings", type=str)
    parser.add_argument('-weight_path', '--weight_path',
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/weights/backbone_ir50_asia.pth", type=str)
    parser.add_argument('-cpu', '--cpu', default=True, type=bool)
    parser.add_argument('-crop_size', '--crop_size', default=112, type=int)
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    embeddings_folder = args.embeddings_folder

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    # extraction feature model
    arcface_r50_asian = IR_50([args.crop_size, args.crop_size])
    arcface_r50_asian.load_state_dict(torch.load(args.weight_path, map_location='cpu' if args.cpu else 'cuda'))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device)

    folders = os.listdir(dataset_folder)
    folders.sort(key=lambda x: int(x.split(".")[0]))

    if not os.path.isdir(embeddings_folder):
        os.mkdir(embeddings_folder)

    for subfolder in tqdm(folders):
        embeddings = list()

        images = os.listdir(os.path.join(dataset_folder, subfolder))
        images.sort(key=lambda x: int(x.split(".")[0]))

        for image_name in images:
            print("Get embedding {}".format(os.path.join(dataset_folder, subfolder, image_name)))
            img = Image.open(os.path.join(dataset_folder, subfolder, image_name))
            img = np.array(img)

            img = preprocess(img, args.cpu)
            embedding = arcface_r50_asian(img)
            embedding = embedding.detach().numpy()
            embeddings.append(l2_normalize(embedding.flatten()))

        embeddings = np.array(embeddings)
        filename = subfolder + ".npz"
        np.savez(os.path.join(embeddings_folder, filename), embeddings)








