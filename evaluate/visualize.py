import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import json

if __name__ == '__main__':
    parser = ArgumentParser("visualize comparison")
    parser.add_argument("-positive_pairs", "--positive_pairs",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/positive_pairs.json",
                        help="specify your json file that contain degree dictionary of positive pairs", type=str)
    parser.add_argument("-negative_pairs", "--negative_pairs",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/negative_pairs.json",
                        help="specify your json file that contain degree dictionary of negative pairs", type=str)
    parser.add_argument("-figure", "--figure",
                        default="/home/phamvanhanh/PycharmProjects/FaceVerification/result/result.png", type=str)
    args = parser.parse_args()

    degrees_dictionary1 = json.loads(open(args.positive_pairs, 'r').read())
    degrees_dictionary2 = json.loads(open(args.negative_pairs, 'r').read())

    fig, ax = plt.subplots(figsize=(12, 5))
    keys_intersection = list()
    keys1 = list(degrees_dictionary1.keys())
    keys1.sort(key=lambda x: int(x))
    data1 = [degrees_dictionary1[key] for key in keys1]

    keys2 = list(degrees_dictionary2.keys())
    keys2.sort(key=lambda x: int(x))
    data2 = [degrees_dictionary2[key] for key in keys2]

    keys_intersection = [key for key in keys1 if key in keys2]
    data_intersection = [min(degrees_dictionary1[key], degrees_dictionary2[key]) for key in keys_intersection]
    print(keys_intersection)
    """    
    for key in keys1:
        print(key, ": ", degrees_dictionary1[key])
        
    for key in keys2:
        print(key, ": ", degrees_dictionary2[key])
        """
    plt.bar(keys1, data1, color='#0047ab', label='positive_pairs')
    plt.bar(keys2, data2, color='#FF0000', label='negative_pairs')
    plt.bar(keys_intersection, data_intersection, color='#802456')

    plt.ylabel('Pair Numbers')
    plt.xlabel('Angles Between Positive and Negative Pairs')
    plt.xticks(np.arange(0, 110, 10))
    plt.legend()
    plt.savefig(args.figure)


