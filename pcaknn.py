# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import operator

global redEigVects

def pca(dataMat, topNfeat=9999999):
    global redEigVects
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)#sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]   #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
def KNN(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=0, 表示列。axis=1, 表示行。
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

filename = '/Users/mac/Downloads/train-images-idx3-ubyte'
binfile = open(filename , 'rb')
buf = binfile.read()
index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
index += struct.calcsize('>IIII')
'''''
# 输出大端数
print magic
print numImages
print numRows
print numColumns
'''

filename1 =  '/Users/mac/Downloads/train-labels-idx1-ubyte'
binfile1 = open(filename1 , 'rb')
buf1 = binfile1.read()

index1 = 0
magic1, numLabels1 = struct.unpack_from('>II' , buf , index)
index1 += struct.calcsize('>II')


#设置训练数目为2500个
trainingNumbers=60000
#降维后的维度为784个维度　降维后的数据为40维度
DD=49 * 7
#初始化traingMat
trainingMatO=zeros((trainingNumbers,28*28))
#初始化标签
trainingLabels=[]


#获取经过PCA  处理过的traingMat 和 label
#for i in range(trainingNumbers):
for i in range(trainingNumbers):
    im = struct.unpack_from('>784B' ,buf, index)
    index += struct.calcsize('>784B')
    im = np.array(im)
    trainingMatO[i]=im
    #读取标签
    numtemp = struct.unpack_from('1B' ,buf1, index1)
    label = numtemp[0]
    index1 += struct.calcsize('1B')
    trainingLabels.append(label)

#PCA
trainingMat,reconMat=pca(trainingMatO,DD)

#eigendigits
fig = plt.figure()
for i in range(0,9):
    im = reconMat[i]
    im = np.array(im)
    im = im.reshape(28,28)
    im = np.real(im)
    plotwindow = fig.add_subplot('33%d'%((i+1)%9))
    plt.imshow(im , cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
#plt.title('D = %d' % DD)
plt.show()
'''''
**************************************************
'''
# 读取testMat
filename3 = '/Users/mac/Downloads/t10k-images-idx3-ubyte'
binfile3 = open(filename3 , 'rb')
buf3 = binfile3.read()
index3 = 0
#'>IIII'使用大端法读取四个unsigned int32
magic3, numImages3 , numRows3 , numColumns3 = struct.unpack_from('>IIII' , buf3 , index3)
index3 += struct.calcsize('>IIII')

#读取labels
filename4 =  '/Users/mac/Downloads/t10k-labels-idx1-ubyte'
binfile4 = open(filename4, 'rb')
buf4 = binfile4.read()

index4= 0
#'>IIII'使用大端法读取两个unsigned int32
magic4, numLabels4 = struct.unpack_from('>II' , buf4 , index4)
index4 += struct.calcsize('>II')

'''''
**************************************************
'''
#测试数据
fig = plt.figure()
testNumbers=520
#测试维度
errCount=0
#获取经过PCA  处理过的testMat 和 label
errImage=0
'''
for i in range(testNumbers):
    im3 = struct.unpack_from('>784B' ,buf3, index3)
    index3 += struct.calcsize('>784B')
    im3 = np.array(im3)

    #新进来的数据　进行降维处理
    meanVals = mean(im3, axis=0)
    meanRemoved = im3 - meanVals #remove mean
    #这个时候使用的降维特征变量为上边给训练数组得出的特征量
    testingMat=meanRemoved*redEigVects
    # testingMat = im3
    #读取标签
    numtemp4 = struct.unpack_from('1B' ,buf4, index4)
    label4 = numtemp4[0]
    index4 += struct.calcsize('1B')
    #.getA() 函数的意思是　获取该矩阵　好像PCA算法返回的是一个对象　所以此处提取了一下矩阵数组
    classifierResult = KNN(testingMat.getA(), trainingMat.getA(), trainingLabels, 17)
    #print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, label4)
    if  classifierResult is not label4:
        print "error class is: %d, the real class is: %d" % (classifierResult, label4)
        errim=im3.reshape(28,28)
        errImage += 1
        ax = fig.add_subplot(5,5,errImage)
        ax.set_title("error class: %d,real class: %d" % (classifierResult, label4))
        plt.imshow(errim)
        plt.xticks([])
        plt.yticks([])
        errCount=errCount+1
plt.show()
print 'the total number of training images is', trainingNumbers
print 'the total number of testing images is', testNumbers
print 'the err rate is ',float(errCount)/testNumbers
'''
