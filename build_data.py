import os
import sys
import getopt
from preprocessing.parse_annotation import parse_label_img, parse_naver_json
from preprocessing.create_map import build_affinity_map, build_regional_map

WIDTH = 512
HEIGHT = 512


def main(argv):
    # build_data.py data-directory data-format char
    print(argv)

    assert argv[2] in ['naver-cord', 'labelImg'], "Not correct data format"
    assert argv[3] in ['char', 'word'], "Box Unit as to be one of char or word"

    base_route = os.getcwd()

    # File always have to be in a folder or train, test folder

    # 1. find correct data directory
    try:
        img_route = base_route + '/data/' + argv[1] + '/train/image/'
        _ = os.listdir(img_route)

    except FileNotFoundError:
        img_route = base_route + '/data/' + argv[1] + '/image/'
        _ = os.listdir(img_route)

    # 2. Parsing Annotation file
    if argv[2] == 'naver-cord':
        parsed_data = parse_naver_json()
    elif argv[2] == 'labelImg':
        parsed_data = parse_label_img()
    else:
        raise ValueError("Annotation file format has to be one of 'naver-cord' or 'labelImg' ")

    # 3. Setting on map, word can figure out affinity map
    if argv[3].lower() == 'char':
        build_regional_map()
    elif argv[3].lower() == 'word':
        build_regional_map()
        build_affinity_map()
    else:
        raise ValueError("Data Unit size has to be one of 'char' or 'word'")


if __name__ == "__main__":
    main(sys.argv)
