import os
import sys
import getopt
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

from model.craft_detect_model import Craft
import config


def main(argv):
    assert argv[2] in ['char', 'word'], "Need To Choose 'char' or 'word'"

    # Set config, directory
    config.init()
    num_map = 1 if argv[2] == 'char' else 2
    base_route = os.getcwd()
    img_route = base_route + '/data/' + argv[1] + '/resized_img/'
    map_route = base_route + '/data/' + argv[1] + '/map/'

    # Load image data
    img_data = [cv2.imread(img_route + x).astype(np.float32) for x in os.listdir(img_route)]

    # Load Region map
    region_data = [pickle.load(open(map_route + x, 'rb')).astype(np.float32) / 255 for x in
                   os.listdir(map_route) if '_region' in x]

    # Load Affinity map
    affinity_data = [pickle.load(open(map_route + x, 'rb')).astype(np.float32) / 255 for x in
                     os.listdir(map_route) if '_affinity' in x]

    # data split
    train_x, test_x, train_y, test_y = train_test_split(img_data, region_data,
                                                        test_size=0.3, random_state=42)

    # Create Pre-trained Directory
    i = 0
    while True:
        try:
            os.mkdir('pre_trained/' + argv[1] + '_' + str(i))
            break
        except OSError:
            i += 1

    save_route = 'pre_trained/' + argv[1] + '_' + str(i) + '/'

    # init model
    craft_model = Craft(config=config, map_num=num_map)
    craft_model.build((None, config.width, config.height, 3))

    # Train Model
    craft_model.compile_model()

    craft_model.train_model(np.asarray(train_x), np.asarray(train_y), np.asarray(test_x), np.asarray(test_y),
                            save_route=save_route)

    # Save Model
    craft_model.save_weights(save_route + 'last_model', overwrite=False)


if __name__ == "__main__":
    # main(sys.argv)
    main(['train.py', 'naver-cord', 'char'])
