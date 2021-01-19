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

def get_lable(label_file, name):

    with open(label_file, 'r') as file:
        for line in file.readlines():
            n, l = line.split(" ")[0:2]
            if name == n:
                la = int(l.split("\n")[0])
                break

    return la

if __name__ == '__main__':

    parser = ArgumentParser(description="get all embeddings dataset")
    parser.add_argument("-weight_path", "--weight_path", default="./weights/backbone_ir50_asia.pth", type=str)
    parser.add_argument("-cpu", "--cpu", default=True, type=bool)
    parser.add_argument("-crop_size", "--crop_size", default=112, type=int)
    parser.add_argument("-default_root", "--default_root", default= "./dataset/images", type=str)
    parser.add_argument("-label", "--label", default="./dataset/label.txt", type=str)
    parser.add_argument("-name", "--name",default="hanh" ,type=str)
    parser.add_argument("-mode", "--mode", default=2, type=int)
    parser.add_argument("-aligned_dataset", "--aligned_dataset",
                        default='/home/phamvanhanh/PycharmProjects/FaceRecognition/dataset/images', type=str)

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device('cpu' if args.cpu else 'cuda:0')

    # Feature Extraction Model
    arcface_r50_asian = IR_50([args.crop_size, args.crop_size])
    arcface_r50_asian.load_state_dict(torch.load(args.weight_path, map_location='cpu' if args.cpu else 'cuda'))
    arcface_r50_asian.eval()
    arcface_r50_asian.to(device)
    label_file = args.label

    if args.mode==1:
        images = glob.glob(os.path.join(args.default_root, args.name) + "/*.jpg")
        print("len:", len(images))
        try:
            idxs = np.random.randint(0, len(images), 10)
        except Exception as e:
            print(e)
            print("directory contains lower than 10 images")

        label = get_lable(label_file, args.name)

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

    else:
        folders = os.listdir(args.aligned_dataset)
        for folder in folders:
            name = folder.split("/")[-1]
            images = glob.glob(os.path.join(args.aligned_dataset, name) + "/*.jpg")

            print("len:", len(images))

            try:
                idxs = np.random.randint(0, len(images), 10)
            except Exception as e:
                print(e)
                print("directory contains lower than 10 images")

            with open(args.label, 'r') as file:
                for line in file.readlines():
                    idx = [i for i in range(len(line)) if line[i]==" "]

                    n = line[:idx[-1]]
                    l = line.split(" ")[-1]
                    if name == n:
                        la = int(l.split("\n")[0])
                        print(name)
                        break
            print(la)

            embeddings = list()
            labels = np.ones(min(len(images), 10)) * la
            for i in range(min(len(images), 10)):
                img = cv2.imread(images[idxs[i]])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                model_input = preprocess(img, args.cpu)

                embed = arcface_r50_asian(model_input)
                embed = embed.detach().numpy()
                embeddings.append(embed.flatten())

            embeddings = np.array(embeddings)
            print(embeddings.shape)

            np.savez("./dataset/embeddings/" + str(la) + ".npz", embeddings, labels)




