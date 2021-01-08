"""
灰度直方图
"""
from matplotlib import pyplot as plt
import cv2
import numpy as np

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 返回一个ndarray对象
cv2.imshow('gray', gray)

'''
# 灰度直方图 方法1 plt.hist()函数
plt.figure()
plt.hist(gray.ravel(), 256) # ravel()可以将二维图像拉平为一维数组
plt.show()
'''

'''
# 灰度直方图 方法2
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# 新建一个图像
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(hist)
plt.xlim([0, 256])  # 设置x坐标轴范围
plt.show()
'''

# 彩色图像直方图
image = cv2.imread('lenna.png')
chans = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()
