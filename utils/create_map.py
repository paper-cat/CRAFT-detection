import numpy as np
from math import exp, sqrt
import cv2
import pickle
import tqdm

import config


def build_regional_map(img_name: str, data: dict, data_route: str):
    """ make regional map according to meta data

    make regional map and save it in map directory

    Args:
        img_name: img file name
        data: meta data include x1...y4, width, height
        data_route: route to save map
    """
    width = config.map_width
    height = config.map_height

    empty_map = make_empty_map(width, height)
    origin_width = data['width']
    origin_height = data['height']

    for word in data['words']:
        for char in word['chars']:
            x1 = int((char['x1'] / origin_width) * width)  # x min
            x2 = int((char['x2'] / origin_width) * width)  # x max
            y1 = int((char['y1'] / origin_height) * height)  # y min
            y3 = int((char['y3'] / origin_height) * height)  # y max

            char_width = y3 - y1
            char_height = x2 - x1

            gau_map = make_gaussian_map(char_width, char_height)

            try:
                empty_map[y1:y3, x1:x2] = gau_map
            except ValueError:
                print(img_name, " has problem when making regional map")
                print(y1, y3, x1, x2)

    with open(data_route + '/map/' + img_name + '_region.pickle', 'wb') as f:
        pickle.dump(empty_map, f, pickle.HIGHEST_PROTOCOL)


def build_affinity_map(img_name: str, data: dict, data_route: str):
    """ make affinity map according to meta data

    make affinity map and save it in map directory

    Args:
        img_name: img file name
        data: meta data include x1...y4, width, height
        data_route: route to save map
    """

    width = config.map_width
    height = config.map_height

    empty_map = make_empty_map(width, height)
    origin_width = data['width']
    origin_height = data['height']

    # TODO : 기울어진 사각형을 어떻게 그려서 넣지?!?!?!? ... 일단 정형화된 (not tilted) 문서에 관해서만 구현
    for word in data['words']:

        prev_x = None
        prev_y_top = None
        prev_y_bot = None

        for char in word['chars']:
            x1 = int((char['x1'] / origin_width) * width)  # x min
            x2 = int((char['x2'] / origin_width) * width)  # x max
            y1 = int((char['y1'] / origin_height) * height)  # y min
            y3 = int((char['y3'] / origin_height) * height)  # y max

            cur_x = int((x1 + x2) / 2)
            cur_y_top = y3
            cur_y_bot = y1

            if prev_x is not None:
                top_y = max(cur_y_top, prev_y_top)
                bot_y = min(cur_y_bot, prev_y_bot)

                try:
                    gau_map = make_gaussian_map(top_y - bot_y, cur_x - prev_x)
                    empty_map[bot_y:top_y, prev_x:cur_x] = gau_map
                except ValueError:
                    pass

            prev_x = cur_x
            prev_y_top = cur_y_top
            prev_y_bot = cur_y_bot

    with open(data_route + '/map/' + img_name + '_affinity.pickle', 'wb') as f:
        pickle.dump(empty_map, f, pickle.HIGHEST_PROTOCOL)


def build_region_affinity_map(img_name: str, data: dict, data_route: str):
    """ build regional and affinity map

    call build_regional_map to make and save regional map
    call build_affinity_map to make and save affinity map

    Args:
        img_name: img file name
        data: meta data include x1...y4, width, height
        data_route: route to save map

    """

    build_regional_map(img_name=img_name, data=data, data_route=data_route)
    build_affinity_map(img_name=img_name, data=data, data_route=data_route)


def save_resized_img(img_name: str, img_route: str, data_route: str):
    img = cv2.imread(img_route + img_name + '.jpg')
    if img is None:
        img = cv2.imread(img_route + img_name + '.png')

    if img is None:
        raise FileNotFoundError

    img = cv2.resize(img, (config.width, config.height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(data_route + '/resized_img/' + img_name + '.jpg', img)


def make_gaussian_map(width, height):
    empty_map = np.zeros((width, height), np.int32)
    for i in range(width):
        for j in range(height):
            dist_x = 2.5 * np.linalg.norm(np.array(i - width / 2)) / (width / 2)
            dist_y = 2.5 * np.linalg.norm(np.array(j - height / 2)) / (height / 2)

            scaled_gaussian_prob = exp(-((dist_x ** 2) / 2 + (dist_y ** 2) / 2))
            empty_map[i, j] = np.clip(scaled_gaussian_prob * 255, 0, 255)

    gaussian_map = empty_map

    return gaussian_map


def make_empty_map(width, height):
    empty_map = np.zeros((width, height), np.int32)
    return empty_map
