from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser(description="compare embeddings for evaluate model")
    parser.add_argument('-embeddings_folder', '--embeddings_folder',
                         default="/home/phamvanhanh/PycharmProjects/FaceVerification/embeddings", type=str)
    parser.add_argument('-compare_positivepairs', '--compare_positivepairs', default=False, type=bool)
    parser.add_argument('-result_file', '--result_file', default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/negative_pairs.json",
                        help='write result of comparison to file for visualize', type=str)
    args = parser.parse_args()

    embeddings_folder = args.embeddings_folder
    degrees_dictionary = dict()

    if args.compare_positivepairs:

        for filename in tqdm(os.listdir(embeddings_folder)):
            embeddings = np.load(os.path.join(embeddings_folder, filename))['arr_0']

            cosins = cosine_similarity(embeddings, embeddings)
            radians = np.arccos(np.clip(cosins, -1, 1))
            degrees = radians / np.pi * 180

            for row in range(degrees.shape[0]):
                for col in range(degrees.shape[1]):
                    if row == col:
                        continue
                    if abs(int(degrees[row, col])) in degrees_dictionary.keys():
                        degrees_dictionary[abs(int(degrees[row, col]))] += 1
                    else:
                        degrees_dictionary[abs(int(degrees[row, col]))] = 1
        keys = list(degrees_dictionary.keys())
        keys.sort()
        total = 0
        for key in keys:
            print(key, ": ", degrees_dictionary[key])
            degrees_dictionary[key] = int(degrees_dictionary[key] / 2)
            total += degrees_dictionary[key]
        print(total)

        with open(args.result_file, 'w') as outfile:
            json.dump(degrees_dictionary, outfile)


    else:
        all_embeddings = list()
        embeddings_files = os.listdir(embeddings_folder)
        embeddings_files.sort()
        for filename in embeddings_files:
            embeddings = np.load(os.path.join(embeddings_folder, filename))['arr_0']
            all_embeddings.append(embeddings)

        all_embeddings = np.array(all_embeddings)

        for i in range(all_embeddings.shape[0]):
            for j in range(all_embeddings.shape[1]):
                ori_embedding = all_embeddings[i, j:j+1]

                # compare ori_embedding with 9 embeddings of other people
                idxes_random = np.random.randint(0, all_embeddings.shape[0], size=9)

                while i in idxes_random:
                    idxes_random = np.random.randint(0, all_embeddings.shape[0], size=9)

                temp_embeddings = list()
                for x, y in zip(idxes_random, np.random.randint(9, size=9)):
                    temp_embeddings.append(all_embeddings[x, y])

                temp_embeddings = np.array(temp_embeddings)
                cosins = cosine_similarity(ori_embedding, temp_embeddings)
                radians = np.arccos(cosins)
                degrees = radians / np.pi * 180
                degrees = degrees.flatten()

                for degree in degrees:
                    if abs(int(degree)) in degrees_dictionary.keys():
                        degrees_dictionary[abs(int(degree))] += 1
                    else:
                        degrees_dictionary[abs(int(degree))] = 1

        keys = list(degrees_dictionary.keys())
        keys.sort()
        total = 0
        for key in keys:
            print(key, ": ", degrees_dictionary[key])
            total += degrees_dictionary[key]
        print(total)

        with open(args.result_file, 'w') as outfile:
            json.dump(degrees_dictionary, outfile)





