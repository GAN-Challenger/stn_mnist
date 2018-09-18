
#coding:utf-8
import layers
import tensorflow as tf

def net(name,image,output0,reuse = None):
    with tf.variable_scope(name,reuse=reuse):
        #conv bias pool 20
        params = []

        conv1,k1 = layers.conv(name + "conv1",image,3,3,1,1,"VALID",1,16)
        bias1,b1 = layers.bias(name + "bias1",conv1,16)
        relu1 = layers.relu(name + "relu1",bias1)
        pool1 = layers.pooling(name + "pool1",relu1,2,2,2,2)
        params += [k1,b1]
		#conv bias pool 10
        conv2,k2 = layers.conv(name + "conv2",pool1,3,3,1,1, "SAME",16,32)
        bias2,b2 = layers.bias(name + "bias2",conv2,32)
        relu2 = layers.relu(name + "relu2",bias2)
        pool2 = layers.pooling(name + "pool2",relu2,2,2,2,2)
        params += [k2,b2]
		#conv bias pool 5
        conv3,k3 = layers.conv(name + "conv3",pool2,3,3,1,1,"SAME",32,64)
        bias3,b3 = layers.bias(name + "bias3",conv3,64)
        relu3 = layers.relu(name + "relu3",bias3)
        pool3 = layers.pooling(name + "pool3",relu3,2,2,2,2)
        params += [k3,b3]
        #conv4
        conv4,k4 = layers.conv(name + "conv4",pool3,3,3,1,1,"VALID",64,128)
        bias4,b4 = layers.bias(name + "bias4",conv4,128)
        relu4 = layers.relu(name + "relu4",bias4)
        params += [k4,b4]
        #conv5
        conv5,k5 = layers.conv(name + "conv5",relu4,3,3,1,1,"VALID",128,128)
        bias5,b5 = layers.bias(name + "bias5",conv5,128)
        relu5 = layers.relu(name + "relu5",bias5)
        params += [k5,b5]
        #fcn
        feature0,dim0 = layers.reshapeToLine(relu5)
        fcn1,k6  = layers.fcn(name + "fcn1",feature0,dim0,output0)
        fcn1_bias,b6 = layers.bias(name + "fcn1_bias",fcn1,output0)
        params += [k6,b6]

        return fcn1_bias,params

