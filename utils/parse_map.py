from copy import deepcopy
import numpy as np
import math
from PIL import Image, ImageDraw
import cv2
import config


def parse_region_map(reg_map, min_heat, min_char_size):
    # 큰 값을 가지는, 중심점이 될 후보들을 찾는다
    center_poses = []
    for x_idx, y_axis in enumerate(reg_map):
        for y_idx, x_value in enumerate(y_axis):
            if reg_map[x_idx][y_idx] > min_heat:

                # 중심점 후보들의 거리가 가까우면, 둘중 큰 값만 남긴다
                if len(center_poses) == 0:
                    center_poses.append((x_idx, y_idx))
                else:
                    checked = False
                    for pre_saved in center_poses:
                        if math.sqrt((pre_saved[0] - x_idx) ** 2 + (pre_saved[1] - y_idx) ** 2) < min_char_size:
                            checked = True
                            if reg_map[x_idx][y_idx] > reg_map[pre_saved[0]][pre_saved[1]]:
                                center_poses.remove(pre_saved)
                                center_poses.append((x_idx, y_idx))

                            else:
                                break
                    if checked is False:
                        center_poses.append((x_idx, y_idx))
    # Box 그리기
    boxes = []
    for center in center_poses:
        cur_x = center[0]
        cur_y = center[1]

        width_left, width_right, height_up, height_down = 1, 1, 1, 1

        while True:
            if reg_map[cur_x - width_left][cur_y] < reg_map[cur_x - width_left + 1][cur_y]:
                width_left += 1
            else:
                break

        while True:
            if reg_map[cur_x + width_right][cur_y] < reg_map[cur_x + width_right - 1][cur_y]:
                width_right += 1
            else:
                break

        while True:
            if reg_map[cur_x][cur_y + height_up] < reg_map[cur_x][cur_y + height_up - 1]:
                height_up += 1
            else:
                break

        while True:
            if reg_map[cur_x][cur_y - height_down] < reg_map[cur_x][cur_y - height_down + 1]:
                height_down += 1
            else:
                break

        boxes.append([cur_x - width_left, cur_y - height_down, cur_x + width_right, cur_y + height_up])

    # Box 들의 겹치는 영역 많으면 하나 제거
    unique_boxes = deepcopy(boxes)
    overlapped = []
    for i, unique_box in enumerate(boxes):
        for j, box in enumerate(boxes):
            if j <= i:
                pass
            else:
                x_right = min(unique_box[2], box[2])
                x_left = max(unique_box[0], box[0])
                y_top = min(unique_box[3], box[3])
                y_bot = max(unique_box[1], box[1])

                if x_right <= x_left:
                    pass
                elif y_top <= y_bot:
                    pass
                else:
                    unique_area = (unique_box[2] - unique_box[0]) * (unique_box[3] - unique_box[1])
                    overlap_area = (x_right - x_left) * (y_top - y_bot)

                    ratio = float(overlap_area / unique_area)
                    if i != j:
                        if 1.0 >= ratio > 0.8:
                            try:
                                unique_boxes.remove(box)
                                overlapped.append(box)
                            except:
                                pass

    print(len(overlapped), "is overlapped")
    return boxes


def parse_reg_aff_map(reg_map, aff_map, min_heat, min_char_size):
    region_boxes = parse_region_map(reg_map, min_heat, min_char_size)
    aff_boxes = parse_region_map(aff_map, min_heat, min_char_size)

    word_boxes = []

    padding = 5

    for aff_box in aff_boxes:
        x_min = aff_box[0]
        y_min = aff_box[1]
        x_max = aff_box[2]
        y_max = aff_box[3]

        regs_in_aff = []
        for reg_box in region_boxes:
            if (x_min < (reg_box[0]+reg_box[2])/2 < x_max) and (y_min-padding < reg_box[1] < y_max+padding):
            # if (reg_box[0] < x_min < reg_box[2]) and (reg_box[1]-padding < (y_min + y_max) / 2 < reg_box[3]+padding):
                regs_in_aff.append(reg_box)

            elif (x_min < (reg_box[0]+reg_box[2])/2 < x_max) and (y_min-padding < reg_box[3] < y_max+padding):
            # elif (reg_box[0] < x_max < reg_box[2]) and (reg_box[1]-padding < (y_min + y_max) / 2 < reg_box[3]+padding):
                regs_in_aff.append(reg_box)

        for i, reg in enumerate(regs_in_aff):
            found = False
            for word in word_boxes:
                if reg in word:
                    found = True
                    for temp in regs_in_aff:
                        word.append(temp)
                    break

            if found is False and i == 0:
                word_boxes.append(regs_in_aff)
                break
            elif found is False and i != 0:
                word_boxes.append([reg])

    for i, word in enumerate(word_boxes):
        word_boxes[i] = list(set(tuple(temp) for temp in word))

    combined_words = []

    for word in word_boxes:
        x_min, y_min, x_max, y_max = 100000, 1000000, 0, 0

        for char in word:
            x_min = char[0] if x_min > char[0] else x_min
            x_max = char[2] if x_max < char[2] else x_max
            y_min = char[1] if y_min > char[1] else y_min
            y_max = char[3] if y_max < char[3] else y_max

        combined_words.append((x_min, y_min, x_max, y_max))

        # if len(combined_words) > 10:
        #     break

    return combined_words


def draw_box(img, boxes):
    config.init()
    img = cv2.resize(img, (config.map_width, config.map_height))
    for box in boxes:
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255))

    return img
