
#coding:utf-8
import struct
import numpy as np

_tag = '>' #ʹ�ô�˶�ȡ
_twoBytes = 'II' #��ȡ���ݸ�ʽ����������
_fourBytes =  'IIII' #��ȡ�����ݸ�ʽ���ĸ�����
_pictureBytes =  '784B' #��ȡ��ͼƬ�����ݸ�ʽ��784���ֽڣ�28*28
_lableByte = '1B' #��ǩ��1���ֽ�
_msb_twoBytes = _tag + _twoBytes
_msb_fourBytes = _tag +  _fourBytes
_msb_pictureBytes = _tag + _pictureBytes
_msb_lableByte = _tag + _lableByte

def getImage(filename = None):
    binfile = open(filename, 'rb') #�Զ����ƶ�ȡ�ķ�ʽ���ļ�
    buf = binfile.read() #��ȡ�ļ����ݻ�����
    binfile.close()
    index = 0 #ƫ����
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
    buf = binfile.read() #��ȡ�ļ����ݻ�����
    binfile.close()
    index = 0 #ƫ����
    numMagic, numItems = struct.unpack_from(_msb_twoBytes,buf, index)
    index += struct.calcsize(_twoBytes)
    labels = []
    for i in range(numItems):
        value = struct.unpack_from(_msb_lableByte, buf, index)
        index += struct.calcsize(_lableByte)
        labels.append(value[0]) #��ȡֵ������
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