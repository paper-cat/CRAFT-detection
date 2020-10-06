import os
import sys
import getopt
import cv2
import numpy as np

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
    img_data = [np.asarray(cv2.imread(img_route + x)).astype(np.float32) for x in os.listdir(img_route)]

    # Load Region map
    region_data = [np.asarray(cv2.imread(img_route + x)).astype(np.float32) for x in os.listdir(map_route) if
                   '_region' in x]

    # Load Affinity map
    affinity_data = [np.asarray(cv2.imread(img_route + x)).astype(np.float32) for x in os.listdir(map_route) if
                     '_affinity' in x]

    # init model
    craft_model = Craft(num_map)
    craft_model.build((None, config.width, config.height, 3))

    # Train Model
    craft_model.compile_model()
    craft_model.fit(np.asarray(img_data), np.asarray(region_data), epochs=10, batch_size=1)

    # Save Model


if __name__ == "__main__":
    # main(sys.argv)
    main(['train.py', 'naver-cord', 'char'])
