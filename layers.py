
#encoding: utf-8

import tensorflow as tf

def initializeVariable(name,shape,initializer,trainable = True):

    var = tf.get_variable(name, shape, initializer=initializer,trainable = trainable)

    return var

def InitializeVariableWithDecay(name,shape,stddev,weight_decay,trainable = True):

    var = initializeVariable(name,shape,tf.truncated_normal_initializer(stddev=stddev),trainable)

    weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay,name='weight_loss')

    tf.add_to_collection('losses', weight_decay)

    return var

def conv(name,input,kernel_height,kernel_width,\
    stride_vertical,stride_horizon,padding,\
    input_featuremaps,output_featuremaps,trainable = True):

    kernel = InitializeVariableWithDecay(name, \
    [kernel_height, kernel_width,input_featuremaps, output_featuremaps], \
    1e-1, 0,trainable)

    conv = tf.nn.conv2d(input, kernel, \
    [1, stride_vertical, stride_horizon,1], padding=padding)

    return conv,kernel
    
def bias(name,input,featuremaps,trainable = True):

    biases = initializeVariable(name,[featuremaps],\
    tf.constant_initializer(0.0),trainable)

    output = tf.nn.bias_add(input,biases)

    return output,biases

def pooling(name,input,\
    pooling_height,poooling_width,\
    stride_vertical,stride_horizon):
    
    output = tf.nn.max_pool(input, \
    ksize=[1, pooling_height, poooling_width, 1],\
    strides=[1, stride_vertical, stride_horizon, 1],\
    padding='SAME', name=name)
    
    return output

def tanh(name,input):
    return tf.tanh(input,name)

def logits(name,input):
    return tf.sigmoid(input,name)

def relu(name,input):
    return tf.nn.relu(input,name)

def softmax(name,input):
    return tf.nn.softmax(input,-1,name)

def active(name,input,mode):
    
    if(mode == "relu"):
        return relu(name,input)
    elif (mode == "softmax"):
        return softmax(name,input)
    else:
        return input
    
def reshapeToLine(input):

    dim = 1
    list = input.get_shape().as_list()
    for d in range(len(list)-1):
        dim *= list[d+1]
    #reshape
    output = tf.reshape(input,[-1,dim])
    
    return output,dim


def fcn(name,input,input_nodes,output_nodes,trainable = True):

    weight = InitializeVariableWithDecay(name,[input_nodes,output_nodes],1e-2,0,trainable)
    #list = input.get_shape().as_list()
    
    #if(len(list) > 2):
        #input = reshapeToLine(input)
    
    output = tf.matmul(input, weight)

    return output,weight

def crossentropy(name,input,labels,batch_size,nodes):

    sparse_labels = tf.reshape(labels, [batch_size, 1])

    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
 
    concated = tf.concat([indices, sparse_labels],1)

    dense_labels = tf.sparse_to_dense(concated,[batch_size, nodes],1.0, 0.0)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = dense_labels,logits = input, name=name)

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss'),cross_entropy_mean

