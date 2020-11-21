#!/usr/bin/env python
# coding: utf-8

# In[38]:


from numpy import * 
from os import listdir
from collections import Counter


def img2vector(filename): #'trainingDigits/0_0.txt'
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
        lineStr = fr.readline() #读取每一行 '00000000000001111000000000000000\n'
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


#classify(vectorUnderTest, trainingMat, hwLabels, 1)
#trainingMat 训练的特征(1934, 1024)，hwLabels分类id

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
    #tile(inX, (dataSetSize, 1)).shape=(1934, 1024) 广播，复制1934
    dataSetSize = dataSet.shape[0] #1934
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # [1,2,3] [[1,2,3],[4,5,6]]
    # [[1,2,3],[1,2,3]] [[1,2,3],[4,5,6]]
    # 取平方
    sqDiffMat = diffMat ** 2 #(1934, 1024)
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1) #按行相加 (1934,)
    # 开方
    distances = sqDistances ** 0.5 #(1934,) array([12.40967365, 11.91637529,  9.43398113,
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3;x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sortedDistIndicies = distances.argsort() #索引排序，从小到大
    # print 'distances.argsort()=', sortedDistIndicies
    # [3,2,5],[2,3,5] -> [1,0,2]

    # 2. 选择距离最小的k个点
    classCount = {} #{0: 1}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]] #k有多少，就取几个最小的数据
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #统计每个分明类对应的个数
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # return sortedClassCount[0][0]
    # 3.利用max函数直接返回字典中value最大的key
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount #返回每条测试数据属于那个分类

  
def handwritingClassTest():
    # 1. 导入数据
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    #['0_0.txt', '0_1.txt', '0_10.txt', '0_100.txt',
    m = len(trainingFileList)  #1934条数据
    trainingMat = zeros((m, 1024)) #（1934，1024）维0向量
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量 32*32=1024
    for i in range(m):
        fileNameStr = trainingFileList[i] #'0_0.txt'
        fileStr = fileNameStr.split('.')[0]  # take off .txt  #0_0
        # label + _ + num.txt
        classNumStr = int(fileStr.split('_')[0]) #拿到类别0
        hwLabels.append(classNumStr) #len(hwLabels)=1934 每条数据的分类id
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList) #946个
    for i in range(mTest):
        fileNameStr = testFileList[i] #'0_0.txt' 测试数据
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0]) #分类0 真实值
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) #(1, 1024) 每一条测试数据的特征
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 5) #预测值
        #print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()
print(len(listdir('testDigits')))
print(len(listdir('trainingDigits')))







