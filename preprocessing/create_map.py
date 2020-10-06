import numpy as np
from math import exp, sqrt
import cv2
import pickle

import config


def build_regional_map(img_name: str, data: dict, data_route: str):
    width = config.map_width
    height = config.map_height

    empty_map = make_empty_map(width, height)
    origin_width = data['width']
    origin_height = data['height']

    for char in data['chars']:
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

    return 0


def build_region_affinity_map(img_name: str, data: dict, data_route: str):
    build_regional_map(img_name=img_name, data=data, data_route=data_route)
    return 0


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
