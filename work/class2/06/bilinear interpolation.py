"""
双线插值
"""

import cv2
import numpy as np


# def get_target(img, width, height):
#     src_w, src_h, channel = img.shape
#     if src_h == height and src_w == width:
#         return img.copy()
#     target = np.zeros((height, width, channel), dtype=np.uint8)
#     x_rate = float(src_w) / width
#     y_rate = float(src_h) / height
#     for i in range(channel):
#         for dest_y in range(height):
#             for dest_x in range(width):
#                 # 目标在源上的坐标
#                 src_x = (dest_x + 0.5) * x_rate - 0.5
#                 src_y = (dest_y + 0.5) * y_rate - 0.5
#
#                 # 找出用于插值的邻近点
#                 src_x0 = int(np.floor(src_x))
#                 src_x1 = min(src_x0 + 1, src_w - 1)
#                 src_y0 = int(np.floor(src_y))
#                 src_y1 = min(src_y0 + 1, src_h - 1)
#
#                 # img[src_y, src_x, i] x和y调转是因为图片是在x方向赋值，y不变
#                 temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x0, i]
#                 temp1 = (src_x1 - src_x) * img[src_y0, src_x1, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
#                 target[dest_y, dest_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
#
#     return target
#
#
# img = cv2.imread('lenna.png')
# target = get_target(img, 600, 700)
# cv2.imshow('dest', target)
# cv2.waitKey()


def get_target(img, width, height):
    src_w, src_h, channel = img.shape
    result = np.zeros((height, width, channel), dtype=np.uint8)
    if src_w == width and src_h == height:
        return img.copy()
    scale_w = float(src_w) / width
    scale_h = float(src_h) / height
    for i in range(channel):
        for dest_y in range(height):
            for dest_x in range(width):
                # 计算出目标点在源图的坐标
                src_x = (dest_x + 0.5) * scale_w - 0.5
                src_y = (dest_y + 0.5) * scale_h - 0.5

                # 找出用于插值的邻近点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x0, i]
                temp1 = (src_x1 - src_x) * img[src_y0, src_x1, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                result[dest_y, dest_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return result


img = cv2.imread('aa.jpg')
target = get_target(img, 660, 700)
cv2.imshow('target', target)
cv2.waitKey()
