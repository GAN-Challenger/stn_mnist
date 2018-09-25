
#coding:utf-8
import layers
import tensorflow as tf

def net(name,image,output0,trainable = True,reuse = None):
    with tf.variable_scope(name,reuse=reuse):
        params = []
        #conv bias 42->40
        conv1,k1 = layers.conv(name + "conv1",image,3,3,1,1,"VALID",1,16,trainable)
        bias1,b1 = layers.bias(name + "bias1",conv1,16,trainable)
        relu1 = layers.relu(name + "relu1",bias1)
        params += [k1,b1]
        #pool 40->20
        pool1 = layers.pooling(name + "pool1",relu1,2,2,2,2)
	#conv bias 20->20
        conv2,k2 = layers.conv(name + "conv2",pool1,3,3,1,1, "SAME",16,32,trainable)
        bias2,b2 = layers.bias(name + "bias2",conv2,32,trainable)
        relu2 = layers.relu(name + "relu2",bias2)
        params += [k2,b2]
        #conv bias 20->20
        conv2_,k2_ = layers.conv(name + "conv2_",relu2,3,3,1,1, "SAME",32,32,trainable)
        bias2_,b2_ = layers.bias(name + "bias2_",conv2_,32,trainable)
        relu2_ = layers.relu(name + "relu2_",bias2_)
        params += [k2_,b2_]
        #pool 20->10
        pool2 = layers.pooling(name + "pool2",relu2_,2,2,2,2)
	#conv bias 10->10
        conv3,k3 = layers.conv(name + "conv3",pool2,3,3,1,1,"SAME",32,64,trainable)
        bias3,b3 = layers.bias(name + "bias3",conv3,64,trainable)
        relu3 = layers.relu(name + "relu3",bias3)
        params += [k3,b3]
        #conv bias 10->10
        conv3_,k3_ = layers.conv(name + "conv3_",relu3,3,3,1,1,"SAME",64,64,trainable)
        bias3_,b3_ = layers.bias(name + "bias3_",conv3_,64,trainable)
        relu3_ = layers.relu(name + "relu3_",bias3_)
        params += [k3_,b3_]
        #pool 10->5
        pool3 = layers.pooling(name + "pool3",relu3_,2,2,2,2)
        #conv4 5->3
        conv4,k4 = layers.conv(name + "conv4",pool3,3,3,1,1,"VALID",64,128,trainable)
        bias4,b4 = layers.bias(name + "bias4",conv4,128,trainable)
        relu4 = layers.relu(name + "relu4",bias4)
        params += [k4,b4]
        #conv5 3->1
        conv5,k5 = layers.conv(name + "conv5",relu4,3,3,1,1,"VALID",128,128,trainable)
        bias5,b5 = layers.bias(name + "bias5",conv5,128,trainable)
        relu5 = layers.relu(name + "relu5",bias5)
        params += [k5,b5]
        #fcn
        feature0,dim0 = layers.reshapeToLine(relu5)
        fcn1,k6  = layers.fcn(name + "fcn1",feature0,dim0,output0,trainable)
        fcn1_bias,b6 = layers.bias(name + "fcn1_bias",fcn1,output0,trainable)
        params += [k6,b6]

        return fcn1_bias,params

