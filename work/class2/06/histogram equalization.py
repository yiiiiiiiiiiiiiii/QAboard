"""
直方图均衡化
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("target", result)
cv2.waitKey(0)
