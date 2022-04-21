
import matplotlib.pyplot as plt
import numpy as np
img = plt.imread('lenna.png')


# #浮点算法
# coefficient = np.array([0.3,0.59,0.11])
# gray = np.dot(img,coefficient)
# plt.imshow(gray,cmap='gray')
# plt.show()


# #整数方法
# coefficient = np.array([30/100,59/100,11/100])
# gray = np.dot(img,coefficient)
# plt.imshow(gray,cmap='gray')
# plt.show()


# #平均值法
# coefficient = np.array([1/3,1/3,1/3])
# gray = np.dot(img,coefficient)
# plt.imshow(gray,cmap='gray')
# plt.show()


#仅取绿色
# coefficient = np.array([0,1,0])
# gray = np.dot(img,coefficient)
#plt.imshow(gray,cmap='gray')
#plt.show()


# #二值化
# coefficient = np.array([0,1,0])
# gray = np.dot(img,coefficient)
# gray[gray <=0.5] =0
# gray[gray >0.5] =1
# plt.imshow(gray,cmap='gray')
# plt.show()

