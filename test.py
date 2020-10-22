import os
import sys
import cv2
import getopt
import numpy as np
import matplotlib.pyplot as plt

from model.craft_detect_model import Craft
import config
from utils.parse_map import parse_region_map, parse_reg_aff_map, draw_box


def main(argv):
    # check argv
    # check pre-trained model / check test_image / check char or word
    assert os.path.exists(argv[0] + '.index') is True, 'Pre-trained model is not in the direction'
    assert os.path.exists(argv[1]) is True, 'Not Correct test data route'
    assert argv[2].lower() in ['word', 'char'], 'Choose one of "word" or "char" to test image'

    config.init()

    test_img = cv2.resize(cv2.imread(argv[1]), (config.width, config.height), interpolation=cv2.INTER_AREA)
    test_img = test_img.astype(np.float32) / 255

    map_num = 2 if 'word' in argv[0] else 1

    test_model = Craft(config=config, map_num=map_num)
    test_model.load_weights(argv[0]).expect_partial()

    result = test_model.predict(np.array([test_img]))

    try:
        reg_map = np.array(result[0, :, :, 0] * 255, np.uint8)
    except IndexError:
        reg_map = np.array(result[0, :, :] * 255, np.uint8)

    if argv[2].lower() == 'char':

        boxes = parse_region_map(reg_map, config.min_heat, min_char_size=10)
        result_img = draw_box(img=test_img, boxes=boxes)

        '''
        aff_map = np.array(result[0, :, :, 1] * 255, np.uint8)
        boxes = parse_region_map(aff_map, config.min_heat, min_char_size=10)
        result_img = draw_box(img=test_img, boxes=boxes)
        '''
    else:
        # Word
        aff_map = np.array(result[0, :, :, 1] * 255, np.uint8)
        result_boxes = parse_reg_aff_map(reg_map, aff_map, config.min_heat, min_char_size=10)
        result_img = draw_box(img=test_img, boxes=result_boxes)

    plt.imshow(result_img)
    plt.show()


if __name__ == "__main__":
    # main(sys.argv)
    # pre-trained_model / test_image_route / char or word
    main(['pre_trained/naver-cord_6_word/last_model',
          'data/naver-cord/resized_img/receipt_00031.jpg',
          'word'])
