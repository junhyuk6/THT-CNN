import numpy as np
import tensorflow as tf
import time
from scipy.io import FortranFile
import random

########################### Model Parameters ######################################################
#input kernel size (e.g. 33x33)
nxl = 16; nxr = 16; nzl = 16; nzr = 16; 
nx = nxl + 1 + nxr; nz = nzr + 1 + nzl; 

#####################################################################################################
######################################## DEEP LEARNING MODEL ########################################

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv_layer(x, W, b, padding):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b
    #return tf.nn.relu(conv)
    return tf.nn.elu(conv)

def convlayer_bn(x, W, padding, phase_train):
    moving_mean = 0.9
    conv =  tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    conv_bn = tf.layers.batch_normalization(conv, momentum=moving_mean, training=phase_train)#, name='conv1_bn')
    return conv_bn

def fc_layer(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID') + b
    return conv

SizeO = tf.placeholder(tf.int32, shape=[])
x = tf.placeholder(tf.float32, [None, None, None, 3])    ##batch, nz, nx, input_maps
y = tf.placeholder(tf.float32, [None, None])             ##batch, output_maps
LearnRate = tf.placeholder(tf.float32, shape=[])
#beta = tf.placeholder(tf.float32, shape=[])
phase_train = tf.placeholder(tf.bool, name='phase_train')

W_conv1 = weight_variable([3, 3, 3, 24])
conv1_bn = convlayer_bn(x       ,W_conv1, 'VALID', phase_train)
h_conv1 = tf.nn.elu(conv1_bn)

W_conv2 = weight_variable([3, 3, 24, 24])
conv2_bn = convlayer_bn(h_conv1 ,W_conv2, 'VALID', phase_train)
h_conv2 = tf.nn.elu(conv2_bn)

W_conv3 = weight_variable([3, 3, 24, 24])
conv3_bn = convlayer_bn(h_conv2 ,W_conv3, 'VALID', phase_train)
h_conv3 = tf.nn.elu(conv3_bn)

W_fc = weight_variable([nz-3*2, nx-3*2, 24, 1])
b_fc1 = bias_variable([1])
h_fc1 = fc_layer(h_conv3 ,W_fc, b_fc1)
    
predict_y = h_fc1
predict_y = tf.reshape(predict_y, [-1, 1])
real_y = tf.reshape(y, [-1, 1])

cost_y = tf.reduce_mean(tf.square(predict_y - real_y)) 

cost_w = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) +
          tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_fc))

cost = cost_y + 0.0001*cost_w
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(LearnRate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(LearnRate).minimize(cost)

saver = tf.train.Saver(max_to_keep=None)

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())