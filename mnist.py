
#coding:utf-8
import struct
import numpy as np

_tag = '>' #使用大端读取
_twoBytes = 'II' #读取数据格式是两个整数
_fourBytes =  'IIII' #读取的数据格式是四个整数
_pictureBytes =  '784B' #读取的图片的数据格式是784个字节，28*28
_lableByte = '1B' #标签是1个字节
_msb_twoBytes = _tag + _twoBytes
_msb_fourBytes = _tag +  _fourBytes
_msb_pictureBytes = _tag + _pictureBytes
_msb_lableByte = _tag + _lableByte

def getImage(filename = None):
    binfile = open(filename, 'rb') #以二进制读取的方式打开文件
    buf = binfile.read() #获取文件内容缓存区
    binfile.close()
    index = 0 #偏移量
    numMagic, numImgs, numRows, numCols = struct.unpack_from(_msb_fourBytes, buf, index)
    index += struct.calcsize(_fourBytes)
    images = []
    for i in xrange(numImgs):
        imgVal  = struct.unpack_from(_msb_pictureBytes, buf, index)
        index += struct.calcsize(_pictureBytes)

        #print len(imgVal)
        #print type(imgVal[0])

        imgVal  = list(imgVal)
        #for j in range(len(imgVal)):
        #   if imgVal[j] > 1:
        #       imgVal[j] = 1
        images.append(imgVal)
    return np.array(images).reshape((numImgs,28,-1)).astype(np.uint8)

def getLabel(filename=None) :
    binfile = open(filename, 'rb')
    buf = binfile.read() #获取文件内容缓存区
    binfile.close()
    index = 0 #偏移量
    numMagic, numItems = struct.unpack_from(_msb_twoBytes,buf, index)
    index += struct.calcsize(_twoBytes)
    labels = []
    for i in range(numItems):
        value = struct.unpack_from(_msb_lableByte, buf, index)
        index += struct.calcsize(_lableByte)
        labels.append(value[0]) #获取值的内容
    return np.array(labels).reshape((numItems,-1)).astype(np.uint8)

""""
import cv2
import random

LIMIT = 90
X = 14
Y = 14
DX = 7
DY = 7
XX = 42
YY = 42

def rotation(image):
    angle = random.uniform(-90,90)
    rot = cv2.getRotationMatrix2D((X,Y),angle,1)
    rot[0][2] += DX
    rot[1][2] += DY
    dst = cv2.warpAffine(image,rot,(XX,YY))
    return dst
"""
'''
images = getImage("train-images.idx3-ubyte")
labels = getLabel("train-labels.idx1-ubyte")

print images.shape,images.dtype
print labels.shape,labels.dtype

import cv2

cv2.imwrite("test.jpg",rotation(images[0]))

print images[0]
print labels[0]
'''