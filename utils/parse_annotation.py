import json


def parse_label_img():
    return 0


def parse_naver_json(img_name: str, img_route: str, ann_route: str):
    with open(ann_route + img_name + '.json') as ann_file:
        ann_data = json.load(ann_file)

    refined_data = dict([('width', ann_data['meta']['image_size']['width']),
                         ('height', ann_data['meta']['image_size']['height'])])

    refined_data['words'] = []

    for box in ann_data['valid_line']:

        words = box['words']

        for word in words:

            word_dict = dict()
            word_dict['text'] = None
            word_dict['chars'] = []

            word_pos = word['quad']
            word_text = str(word['text'])

            char_counts = len(word_text) if len(word_text) > 0 else 1

            # split given word box into char boxes
            char_width = round((int(word_pos['x2']) - int(word_pos['x1'])) / char_counts, 2)

            assert char_width > 0, "Annotation Error, character length cannot be 0 or less"

            word_dict['text'] = word_text
            for i in range(char_counts):
                word_dict['chars'].append({'x1': int(word_pos['x1']) + char_width * i,
                                           'y1': int(word_pos['y1']),
                                           'x2': int(word_pos['x1']) + char_width * (i + 1),
                                           'y2': int(word_pos['y2']),
                                           'x3': int(word_pos['x1']) + char_width * (i + 1),
                                           'y3': int(word_pos['y3']),
                                           'x4': int(word_pos['x1']) + char_width * i,
                                           'y4': int(word_pos['y4'])})

            refined_data['words'].append(word_dict)

    return refined_data
