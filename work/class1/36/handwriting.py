#!/usr/bin/env python
# coding: utf-8

# In[4]:


from numpy import *
from os import listdir
from collections import Counter


# In[5]:


def img2vector(filename):
    """
    将图像数据转换为向量
    :param filename: 图片文件 因为我们的输入数据的图片格式是 32 * 32的
    :return: 一维矩阵
    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# In[6]:


def classify(inX, dataSet, labels, k):
    """
    inx[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3;x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sortedDistIndicies = distances.argsort()
    # print 'distances.argsort()=', sortedDistIndicies

    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount


def handwritingClassTest(trainingFileList, testFileList):
    # 1. 导入数据
    hwLabels = []
    # trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('data/%s' % fileNameStr)

    # 2. 导入测试数据
    # testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    error_file = []
    pre_dict = {}

    for k_size in range(1, 2):
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]  # take off .txt
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = img2vector('data/%s' % fileNameStr)
            classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, k_size)
            # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
            if (classifierResult != classNumStr):
                errorCount += 1.0
                error_file.append(str(classifierResult) + '\t' + fileNameStr)
        print("k_size: %d" % k_size)
        print("the total number of errors is: %d" % errorCount)
        print("the total error rate is: %f" % (errorCount * 100.0 / float(mTest)))
        # print("\n错误文件：\n" + '\n'.join(error_file))
        print("\n" * 2)
        pre_dict[str(errorCount)] = k_size
    print(pre_dict)


import sklearn.naive_bayes as nb


def nb_test(trainingFileList, testFileList):
    # 1. 导入数据
    hwLabels = []
    # trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(int(classNumStr))
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('data/%s' % fileNameStr)

    model_nb = nb.BernoulliNB()
    model_nb.fit(trainingMat, hwLabels)

    m = len(testFileList)
    hwLabels_test = []
    trainingMat_test = zeros((m, 1024))
    for i in range(m):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels_test.append(int(classNumStr))
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat_test[i, :] = img2vector('data/%s' % fileNameStr)

    result_y = model_nb.predict(trainingMat_test)
    result_y_len = len(result_y)
    hwLabels_test_len = len(hwLabels_test)
    if result_y_len != hwLabels_test_len:
        print('*' * 100)
        print('预测失败！')
        return

    test_dict = {}
    result_dict = {}
    result_dict_ok = {}
    result_dict_error = {}
    error_num = 0
    for y1 in range(result_y_len):
        rs_key = str(result_y[y1])
        result_dict[rs_key] = result_dict.get(rs_key, 0) + 1

        test_key = str(hwLabels_test[y1])
        test_dict[test_key] = test_dict.get(test_key, 0) + 1

        if rs_key == test_key:
            result_dict_ok[test_key] = result_dict_ok.get(test_key, 0) + 1
        else:
            result_dict_error[test_key] = result_dict_error.get(test_key, 0) + 1
            error_num += 1

    print('总错数：' + (str(error_num)))
    print('总错误率：' + (str(error_num * 100.0 / result_y_len)))
    # for key, val in test_dict.items():
    #     print('\n目标数字：' + key + '\t ')
    #
    #     num = str(result_dict_error.get(key, 0))
    #     print('错误个数：' + num)
    #
    #     num = str(result_dict.get(key) * 100.0 / val)
    #     print('正确率：' + num)
    #
    #     num = str(result_dict_ok.get(key) * 100.0 / val)
    #     print('召回率：' + num)
    #
    #     num = str(
    #         result_dict_ok.get(key) * 100.0 / result_dict.get(key))
    #     print('精准率：' + num)
    print('=' * 100)


import sklearn.neighbors as knn


def knn_test(trainingFileList, testFileList):
    # 1. 导入数据
    hwLabels = []
    # trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(int(classNumStr))
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('data/%s' % fileNameStr)

    model_knn = knn.KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(trainingMat, hwLabels)

    m = len(testFileList)
    hwLabels_test = []
    trainingMat_test = zeros((m, 1024))
    for i in range(m):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels_test.append(int(classNumStr))
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat_test[i, :] = img2vector('data/%s' % fileNameStr)

    result_y = model_knn.predict(trainingMat_test)
    result_y_len = len(result_y)
    hwLabels_test_len = len(hwLabels_test)
    if result_y_len != hwLabels_test_len:
        print('*' * 100)
        print('预测失败！')
        return

    test_dict = {}
    result_dict = {}
    result_dict_ok = {}
    result_dict_error = {}
    error_num = 0
    for y1 in range(result_y_len):
        rs_key = str(result_y[y1])
        result_dict[rs_key] = result_dict.get(rs_key, 0) + 1

        test_key = str(hwLabels_test[y1])
        test_dict[test_key] = test_dict.get(test_key, 0) + 1

        if rs_key == test_key:
            result_dict_ok[test_key] = result_dict_ok.get(test_key, 0) + 1
        else:
            result_dict_error[test_key] = result_dict_error.get(test_key, 0) + 1
            error_num += 1

    print('总错数：' + (str(error_num)))
    print('总错误率：' + (str(error_num * 100.0 / result_y_len)))


import os

dir_path = os.path.abspath(os.path.curdir) + '/data'
file_list = os.listdir(dir_path)

import random

for ct in range(7):
    trainingFileList = []
    testFileList = []
    for file in file_list:
        n = random.random()
        if n <= 0.7:
            trainingFileList.append(file)
            continue
        testFileList.append(file)
    print(len(trainingFileList))
    print(len(testFileList))
    # 准确率高，性能要差些
    # handwritingClassTest(trainingFileList, testFileList)
    # knn_test(trainingFileList, testFileList)
    # 性能高，准确率要差点
    nb_test(trainingFileList, testFileList)
