import cv2
import matplotlib.pyplot  as plt
from numpy import *
import numpy
'''*************************************histogram**********************************************************'''
def his_img(img):
    H,W =img.shape[:2]
    img_hist = cv2.calcHist([img],[0],None,[256],[0,256])
    init=0
    img_copy=numpy.zeros((H,W),numpy.uint8)

    for i in range(256):
        init += img_hist[i][0]
        x,y =numpy.where(img==i)
        z=numpy.round(numpy.round(init/(H*W),2)*256)-1
        if z<0:
            z=0
        img_copy[x,y]=z
    return img_copy

def display_his(array):
    array_copy=array.copy()
    array_copy=array_copy.flatten()
    plt.hist(array_copy, bins=256)
    plt.show()

img=cv2.imread(r'C:\Users\xiaoguo\Desktop\lenna.png',0)
img1=his_img(img)
img_hist= cv2.calcHist([img],[0],None,[256],[0,256])
display_his(img_hist)
img1_hist=cv2.calcHist([img1],[0],None,[256],[0,256])
display_his(img1_hist)

cv2.imshow('init_img',img)
cv2.waitKey()
cv2.imshow('img',img1)
cv2.waitKey()

'''*************************************PCA**************************************************************'''
def featurenormalize(X):
    n=X.shape[1]
    mu=numpy.zeros([1,n])
    mu=numpy.mean(X,axis=0)
    for i in range(n):
        X[:,i]=X[:,i]-mu[i]
    return X
def PCA(matrix,num):
   matrix_cov=numpy.cov(matrix,rowvar=False)
   eigvals,eigvecs=numpy.linalg.eig(matrix_cov)
   index=argsort(eigvals)
   index=index[::-1]
   index=index[:num]
   eigvecs=eigvecs[:,index]
   cvt_matrix=numpy.dot(matrix,eigvecs)
   return cvt_matrix


a=numpy.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
a=featurenormalize(a)
print(a)
a=PCA(a,2)
print(a)


