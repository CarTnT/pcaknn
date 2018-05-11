import numpy as np  
import struct  
import matplotlib.pyplot as plt  
  
filename = '/Users/mac/Downloads/train-images-idx3-ubyte'  
binfile = open(filename , 'rb')  # python3 'r'  but python2 'rd'  
buf = binfile.read()  
  
fig = plt.figure()  
index = 0
for i in range(0,9):
    index = i*784
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII' , buf , index)  
    index += struct.calcsize('>IIII')  
  
    im = struct.unpack_from('>784B' , buf , index)  
    index += struct.calcsize('>784B')  
  
    
    im = np.array(im)  
    im = im.reshape(28,28)  
    plotwindow = fig.add_subplot('33%d'%((i+1)%9))  
    plt.imshow(im , cmap = 'gray') 
    plt.xticks([])
    plt.yticks([]) 
plt.show() 
