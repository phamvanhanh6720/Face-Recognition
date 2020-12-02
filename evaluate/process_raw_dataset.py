import os
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description="preprocess raw dataset")
    parser.add_argument("-source_root", "--source_root", help="specify your source directory",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/raw_dataset", type=str)
    args = parser.parse_args()

    source_root = args.source_root
    folders = os.listdir(os.path.join(source_root))
    folders.sort()
    """

    # rename folder name and make labels.txt file that contains many pairs <new folder name>:<original name>
    with open(source_root + "/labels.txt", 'w') as file:
        for i in range(len(folders)):
            source = source_root + "/" + folders[i]
            dest = source_root + "/" + str(i)
            file.write(str(i) + ":" + folders[i] + "\n")
            os.rename(source, dest) 
    """

    # rename name of image in each subfolder
    for subfolder in tqdm(folders):
        images = os.listdir(os.path.join(source_root, subfolder))
        images.sort()

        for i in range(len(images)):
            source = os.path.join(source_root, subfolder, images[i])
            dest = os.path.join(source_root, subfolder, str(i) + ".jpg")
            os.rename(source, dest)



