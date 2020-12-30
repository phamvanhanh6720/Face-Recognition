import glob
import os
import numpy as np
from models.model_irse import IR_50
import torch
from argparse import ArgumentParser
import cv2

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


if __name__ == '__main__':

    parser = ArgumentParser(description="get all embeddings of special folder")
    parser.add_argument("-weight_path", "--weight_path", default="./weights/backbone_ir50_asia.pth", type=str)
    parser.add_argument("-cpu", "--cpu", default=True, type=bool)
    parser.add_argument("-crop_size", "--crop_size", default=112, type=int)
    parser.add_argument("-default_root", "--default_root", default= "./dataset/images", type=str)
    parser.add_argument("-label", "--label", default="./dataset/label.txt", type=str)
    parser.add_argument("-name", "--name",default="hanh" ,type=str)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    # Feature Extraction Model
    arcface_r50_asian = IR_50([args.crop_size, args.crop_size])
    arcface_r50_asian.load_state_dict(torch.load(args.weight_path, map_location='cpu' if args.cpu else 'cuda'))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device)

    images = glob.glob(os.path.join(args.default_root, args.name) + "/*.jpg")
    print("len:", len(images))
    try:
        idxs = np.random.randint(0, len(images), 10)
    except Exception as e:
        print(e)
        print("directory contains lower than 10 images")

    with open(args.label, 'r') as file:
        for line in file.readlines():
            name, l = line.split(" ")[0:2]
            if args.name==name:
                label = int(l.split("\n")[0])
                break

    embeddings = list()
    labels = np.ones(min(len(images), 10)) * label
    for i in range(min(len(images), 10)):
        img = cv2.imread(images[idxs[i]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model_input = preprocess(img, args.cpu)

        embed = arcface_r50_asian(model_input)
        embed = embed.detach().numpy()
        embeddings.append(embed.flatten())

    embeddings = np.array(embeddings)
    print(embeddings.shape)

    np.savez("./dataset/embeddings/" + str(label) + ".npz", embeddings, labels)



