# -*- coding: utf-8 -*-     
import struct    
import matplotlib.pyplot as plt    
import operator  
import numpy as np  
  
# 读取trainingMat  
filename = '/Users/mac/Downloads/train-images-idx3-ubyte'    
binfile = open(filename , 'rb')    
buf = binfile.read()    
index = 0    
#'>IIII'使用大端法读取四个unsigned int32    
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)    
index += struct.calcsize('>IIII')    
  
#读取labels  
filename1 =  '/Users/mac/Downloads/train-labels-idx1-ubyte'    
binfile1 = open(filename1 , 'rb')    
buf1 = binfile1.read()    
    
index1 = 0    
#'>IIII'使用大端法读取两个unsigned int32    
magic1, numLabels1 = struct.unpack_from('>II' , buf , index)    
index1 += struct.calcsize('>II')    
  
#获取经过PCA  处理过的traingMat 和 label  
#for i in range(trainingNumbers):   
for i in range(16):     
    im = struct.unpack_from('>784B' ,buf, index)    
    index += struct.calcsize('>784B')    
    im = np.array(im)   
    im=im.reshape(28,28)  
    plt.subplot(4,4,i+1)  
    plt.imshow(im)  
    #读取标签  
    numtemp = struct.unpack_from('1B' ,buf1, index1)   
    label = numtemp[0]  
    index1 += struct.calcsize('1B')  
    print label  
plt.show()  
