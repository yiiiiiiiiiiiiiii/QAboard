"""
    最邻近插值法
"""
import cv2
import numpy as np


# def get_dest_matrix(img, dest_height, dest_width):
#     height, width, channel = img.shape
#     dest_matrix = np.zeros((dest_width, dest_height, channel), np.uint8)
#     # height_rate = dest_height / height
#     height_rate = height / dest_height
#     # width_rate = dest_width / width
#     width_rate = width / dest_width
#     for i in range(dest_height):
#         for j in range(dest_width):
#             srcX = int(i * height_rate)
#             srcY = int(j * width_rate)
#             dest_matrix[i, j] = img[srcX, srcY]
#     return dest_matrix
#
#
# img = cv2.imread('lenna.png')
# dest_matrix = get_dest_matrix(img, 800, 800)
# print(dest_matrix)
# cv2.imshow("nearest interp", dest_matrix)
# cv2.imshow('src_imge', img)
# cv2.waitKey(0)

def get_target_matrix(img, target_width, target_height):
    src_w, src_h, channels = img.shape
    w_rate = src_w / target_width
    h_rate = src_h / target_height
    target_matrix = np.zeros((target_width, target_height,channels), np.uint8)
    for r in range(target_height):
        for c in range(target_width):
            x = int(r * h_rate)
            y = int(c * w_rate)
            target_matrix[r, c] = img[x, y]
    return target_matrix


# 先获取源矩阵
img = cv2.imread('lenna.png')
# 调用获取目标矩阵的函数
target_matrix = get_target_matrix(img, 700, 700)
cv2.imshow('target', target_matrix)
cv2.imshow('src', img)
cv2.waitKey(0)