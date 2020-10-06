import os
import sys
import getopt
import re

from preprocessing.parse_annotation import parse_label_img, parse_naver_json
from preprocessing.create_map import build_region_affinity_map, build_regional_map

WIDTH = 512
HEIGHT = 512


def main(argv):
    # Run Arguments
    # build_data.py / data-directory / data-format / char
    print(argv)

    assert argv[2] in ['naver-cord', 'labelImg'], "Not correct data format"
    assert argv[3] in ['char', 'word'], "Box Unit as to be one of char or word"

    base_route = os.getcwd()

    # File always have to be in a folder or train, test folder

    # 1. find correct data directory
    try:
        img_route = base_route + '/data/' + argv[1] + '/train/image/'
        ann_route = base_route + '/data/' + argv[1] + '/train/annotation/'
        img_list = [re.findall(r"^(.+)(?=\.)", x)[0] for x in os.listdir(img_route)]

    except FileNotFoundError:
        img_route = base_route + '/data/' + argv[1] + '/image/'
        ann_route = base_route + '/data/' + argv[1] + '/annotation/'
        img_list = [re.findall(r"^(.+)(?=\.)", x)[0] for x in os.listdir(img_route)]

    # MAP creation pipeline
    if argv[2] == 'naver-cord':
        parsing = parse_naver_json

    elif argv[2] == 'labelImg':
        parsing = parse_label_img
    else:
        raise ValueError("Annotation file format has to be one of 'naver-cord' or 'labelImg' ")

    if argv[3].lower() == 'char':
        build_map = build_regional_map
    elif argv[3].lower() == 'word':
        build_map = build_region_affinity_map
    else:
        raise ValueError("Data Unit size has to be one of 'char' or 'word'")

    for img in img_list:
        # 2. Parsing Annotation file
        refined_data = parsing(img_name=img, img_route=img_route, ann_route=ann_route)

        # 3. Setting on map, word can figure out affinity map
        build_map(refined_data)


if __name__ == "__main__":
    # main(sys.argv)
    main(['build_data.py', 'naver-cord-small', 'naver-cord', 'char'])
