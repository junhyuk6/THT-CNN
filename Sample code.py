import numpy as np
import tensorflow as tf
import time
from scipy.io import FortranFile
import random

######################################################################################################
########################################## Loaddata ##################################################

#Model Parameters 
#input kernel size (e.g. 33x33)
nxl = 16; nxr = 16; nzl = 16; nzr = 16; 
nx = nxl + 1 + nxr; nz = nzr + 1 + nzl; 

##Data Parameters
#grid size of data field
nxp = 192; nzp = 192;

# the number of training fields
nTrainField = 100; nField = nTrainField;
nData = 100*nxp*nzp

#learning iteration per same learning rate
subEpoch = 100; nStep = 5; totalEpoch = subEpoch * nStep
batch_size = 16
totalBatch = int(nData/batch_size)

#initial learning rate & lr decays to lr*1/5 per N_epoch
iniLR = 0.0005

def LoadTrainData(nField,iniField,intervalField):
    #r.m.s of data (0 : du/dy, 1 : dw/dy, 2 : p)
    Input_std = np.empty([3])
    Input_std[0] = 66.274826292; Input_std[1] = 37.0097954241; Input_std[2] = 1.58318422533;
    Data = np.empty([nField,nzp,nxp,4], dtype=np.float32) # [number of field, z-grids, x-grids, variables(du/dy, dw/dy, p, dT/dy)]
    for j in range(nField) :
        p = iniField + j*intervalField
        Filename = '/DLdata/' + '%05d'%(p)

        f = FortranFile(Filename,'r') 
        insfield = f.read_reals(dtype='float32')
        Data[j,:,:,:] = np.transpose(np.reshape(insfield,(4,nxp,nzp), order='F'))

    for image1 in range(3):
        Data[:,:,:,image1] = Data[:,:,:,image1]/Input_std[image1]
    return Data

TrainData = LoadTrainData(nTrainField,3000,4)
#TrainData = np.empty([nField,nzp,nxp,4], dtype=np.float32)

# Periodic padding of field data
TrainData = np.concatenate((TrainData[:,:,:,:],TrainData[:,0:nz-1,:,:]), axis=1)
TrainData = np.concatenate((TrainData[:,:,:,:],TrainData[:,:,0:nx-1,:]), axis=2)

batch_xs = np.zeros([batch_size, nz, nx, 3], dtype=np.float32)
batch_ys = np.zeros([batch_size, 1, 1, 1], dtype=np.float32)
def GetBatch(Data):
    # random sampling with replacement
    int1 = np.random.randint(nField, size=batch_size)
    int2 = np.random.randint(nzp, size=batch_size)
    int3 = np.random.randint(nxp, size=batch_size)
    for j in range(batch_size):
        batch_xs[j:j+1,0:nz,0:nx,0:3] = Data[int1[j]:int1[j]+1,
                                             int2[j]:int2[j]+nz,
                                             int3[j]:int3[j]+nx,
                                             0:3]
        batch_ys[j:j+1,0:1,0:1,0:1] = Data[int1[j]:int1[j]+1,
                                           int2[j]+(nz-1)//2:int2[j]+(nz-1)//2+1,
                                           int3[j]+(nx-1)//2:int3[j]+(nx-1)//2+1,
                                           3:4]
    return batch_xs, batch_ys

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
y = tf.placeholder(tf.float32, [None, None, None, 1])             ##batch, output_maps
LearnRate = tf.placeholder(tf.float32, shape=[])
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
#W_fc = weight_variable([nzp-3*2, nxp-3*2, 24, 1])
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

###############################################################################################
######################################### Train ###############################################

for i1 in range(nStep):
    ## Change Learningrate
    rate = iniLR * np.power(5.0,-float(i1))
    for i2 in range(subEpoch):
        for i3 in range(totalBatch):
            batch_xs, batch_ys = GetBatch(TrainData)
            _, cost_curr = sess.run([optimizer, cost_y], feed_dict={x: batch_xs, y: batch_ys, LearnRate: rate, phase_train: True})  
