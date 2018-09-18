
import cv2
import random
import mnist
import numpy as np

LIMIT = 90
X = 14
Y = 14
DX = 7
DY = 7
XX = 42
YY = 42

def rotation(image,phase = "TRAIN"):
    angle = 0
    if phase == "TRAIN":
       random.uniform(-1*LIMIT,LIMIT)
       rot = cv2.getRotationMatrix2D((X,Y),angle,1)
       rot[0][2] += DX
       rot[1][2] += DY
       dst = cv2.warpAffine(image,rot,(XX,YY))
       return dst
    else:
       dst = cv2.resize(image,(XX,YY))
       return dst

class DataSet():
    def __init__(self,train_image,train_label,test_image,test_label):
        self.t_images = mnist.getImage(train_image)
        self.t_labels = mnist.getLabel(train_label)
        
        self.test_images = mnist.getImage(test_image)
        self.test_labels = mnist.getLabel(test_label)

        self.index = 0
        self.list = range(self.t_labels.shape[0])
        random.shuffle(self.list)
        
        self.test_index = 0

    def shuffle(self):
        self.index = 0
        #index = range(60000)
        random.shuffle(self.list)

    def getTrainBatch(self,batch,rt="TRAIN"):
        images = []
        labels = []
        for  i in range(batch):
            if self.index >= self.t_labels.shape[0] :
                self.shuffle()
            images += [rotation(self.t_images[self.index],rt)]
            labels += [self.t_labels[self.index]]
            self.index += 1
        return np.array(images).reshape(batch,XX,YY,1),np.array(labels).reshape(batch,1)

    def getTestBatch(self,batch):
        images = []
        labels = []
        for  i in range(batch):
            if self.test_index >= self.test_labels.shape[0]:
                self.test_index = 0

            images += [rotation(self.test_images[self.test_index],"TEST")]
            labels += [self.test_labels[self.test_index]]
            self.test_index += 1
        return np.array(images).reshape(batch,XX,YY,1),np.array(labels).reshape(batch,1)
