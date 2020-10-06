import json


def parse_label_img():
    return 0


def parse_naver_json(img_name: str, img_route: str, ann_route: str):
    with open(ann_route + img_name + '.json') as ann_file:
        ann_data = json.load(ann_file)

    refined_data = dict([('text', None)])
    refined_data['chars'] = []

    for box in ann_data['valid_line']:
        words = box['words']

        for word in words:
            word_pos = word['quad']
            word_text = str(word['text'])

            char_counts = len(word_text) if len(word_text) > 0 else 1

            # split given word box into char boxes
            char_width = round(int(word_pos['x2']) - int(word_pos['x1']) / char_counts, 2)

            assert char_width > 0, "Annotation Error, character length cannot be 0 or less"

            refined_data['text'] = word_text
            for i in range(char_counts):
                refined_data['chars'].append({'x1': int(word_pos['x1']) + char_width * i,
                                              'y1': int(word_pos['y1']),
                                              'x2': int(word_pos['x1']) + char_width * (i + 1),
                                              'y2': int(word_pos['y2']),
                                              'x3': int(word_pos['x1']) + char_width * (i + 1),
                                              'y3': int(word_pos['y3']),
                                              'x4': int(word_pos['x1']) + char_width * i,
                                              'y4': int(word_pos['y4'])})

    return refined_data
