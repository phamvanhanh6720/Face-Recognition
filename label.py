import os
import glob

if __name__=="__main__":
    path ="/home/phamvanhanh/PycharmProjects/FaceRecognition/dataset/images"

    folders= os.listdir(path)
    count = 1
    with open("./dataset/label.txt", 'w') as file:
        for folder in folders:
            name = folder.split("/")[-1]
            file.write(name + " " + str(count) + "\n")
            count +=1