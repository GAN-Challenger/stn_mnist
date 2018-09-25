
import tensorflow as tf
import mnist_rotation
import transformer
import network
import layers
import time
import cv2

BATCH_SIZE = 256
CLASS_SIZE = 10

#set gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def optimize(loss,global_step,sample_numbers,batch_size,var_list):

    num_batches_per_epoch = sample_numbers / batch_size
    
    decay_steps = num_batches_per_epoch*10
    
    lr = tf.train.exponential_decay(0.005,\
    global_step,decay_steps,0,staircase=True)
    
    #tf.summary.scalar('learning_rate', lr)
    
    #opt = tf.train.MomentumOptimizer(lr,0.9).minimize(loss)
    
    opt = tf.train.GradientDescentOptimizer(lr).minimize(loss,var_list=var_list)
    
    return opt

def __train__():
    #data
    dataset = mnist_rotation.DataSet("mnist/train-images.idx3-ubyte","mnist/train-labels.idx1-ubyte","mnist/t10k-images.idx3-ubyte","mnist/t10k-labels.idx1-ubyte")
    X = tf.placeholder(tf.uint8, [None,42,42,1])
    Y = tf.placeholder(tf.uint8,[None,1])

    images = (tf.cast(X,tf.float32) - 128.0)/128.0
    labels = tf.cast(Y,tf.int32)

    stn,stn_params = network.net("stn",images,6)
    #!!!!!!!!!!!!!use tanh as activation
    #stn_ = layers.tanh("stn_tanh",stn)

    #stn
    images_ = transformer.batch_transformer(images,stn,[42,42])

    """
    R = tf.placeholder(tf.uint8,[None,42,42,1])
    RL = tf.placeholder(tf.uint8,[None,1])

    R_ = (tf.cast(R,tf.float32) - 128.0)/128.0
    RL_ = tf.cast(RL,tf.int32)

    IMAGE = tf.concat([images_,R_],0)
    LABEL = tf.concat([labels,RL_],0)
    """
    #cnn
    cnn,cnn_params = network.net("cnn",images_,CLASS_SIZE,trainable = True)
    loss,cross_loss = layers.crossentropy("stn_loss",cnn,labels,BATCH_SIZE,CLASS_SIZE)
    global_step = tf.Variable(0, trainable=False)
    opt = optimize(loss,global_step,60000,BATCH_SIZE,stn_params + cnn_params)
    #cnn
    """_
    cnn_,cnn_params_ = network.net("cnn",images,CLASS_SIZE,trainable = True,reuse = True)
    loss_,cross_loss_ = layers.crossentropy("cnn_loss",cnn_,labels,BATCH_SIZE,CLASS_SIZE)
    global_step_ = tf.Variable(0, trainable=False)
    opt_ = optimize(loss_,global_step_,30000,BATCH_SIZE,cnn_params_)
    #softmax = layers.softmax("cnn_softmax",cnn)
    """
    """
    CNN, CNN_PARAMS = network.net("cnn",IMAGE,CLASS_SIZE,trainable = True,reuse = True:)
    LOSS,CROSS_LOSS = layers.crossentropy("LOSS",CNN,LABEL,BATCH_SIZE*2,CLASS_SIZE)
    GLOBAL_STEP    = tf.Variable(0, trainable=False)
    OPT             = optimize(LOSS,GLOBAL_STEP,60000,BATCH_SIZE*2)
    """

    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)
    stn_saver = tf.train.Saver(stn_params)
    cnn_saver = tf.train.Saver(cnn_params)

    save_file_name = "./transformer_model"
	
    for step in range(100):
        #print "step = ",step
        value = 0
        global_step = step

        start_time = time.time()
        for epoch in range(int(60000/BATCH_SIZE)):
            #print epoch
            #_cnn,_y = sess.run([cnn,Y],feed_dict = {(X,Y):dataset.getTrainBatch(BATCH_SIZE)})
            #print _cnn,_y
            opt_,loss_ = sess.run([opt , cross_loss],feed_dict = {(X,Y):dataset.getTrainBatch(BATCH_SIZE)})
            #value += _loss_value
            print "current loss : ",loss_
            
            #_opt_,_loss_value_ = sess.run([opt_, cross_loss_],feed_dict = {(X,Y):dataset.getTrainBatch(BATCH_SIZE,"TEST")})
            #value += _loss_value
            #print "current loss : ",_loss_value,"  ",_loss_value_
            #_opt,_loss_value = sess.run([OPT , CROSS_LOSS],feed_dict = {(X,Y,R,RL):(dataset.getTrainBatch(BATCH_SIZE),dataset.getTrainBatch(BATCH_SIZE))})
            #value += _loss_value

            #print _loss_value_

        print ("step = ",step)
        print ("loss = ",value/(60000/BATCH_SIZE))
        print ("duration = ",time.time() - start_time)

        #save model
        if (step %1 == 0):
            save_path = stn_saver.save(sess,save_file_name + "_stn")
            print (save_path)

            save_path = cnn_saver.save(sess,save_file_name + "_cnn")
            print (save_path)

#__train__()

def __test__():
    dataset = mnist_rotation.DataSet("mnist/train-images.idx3-ubyte","mnist/train-labels.idx1-ubyte","mnist/t10k-images.idx3-ubyte","mnist/t10k-labels.idx1-ubyte")
    X = tf.placeholder(tf.uint8, [None,42,42,1])
    Y = tf.placeholder(tf.uint8,[None,1])

    images = (tf.cast(X,tf.float32) - 128.0)/128.0
    labels = tf.cast(Y,tf.int32)

    stn,stn_params = network.net("stn",images,6)
    #stn = layers.tanh("stn_tanh",fcn0)

    #stn
    images_ = transformer.batch_transformer(images,stn,[42,42])

    sess = tf.Session()
    stn_saver = tf.train.Saver(stn_params)
    stn_saver.restore(sess,"./transformer_model_stn")

    for i in range(1):
        images,image_ = sess.run([images , images_],feed_dict = {(X,Y):dataset.getTrainBatch(BATCH_SIZE)})
        images = images*128 + 128
        image_ = image_*128 + 128
        for j in range(BATCH_SIZE):
            cv2.imwrite(str(j) + "_src.jpg",images[j])
            cv2.imwrite(str(j) + "_dst.jpg",image_[j])

#with tf.device("cpu/"):
    #__test__()
__train__()
