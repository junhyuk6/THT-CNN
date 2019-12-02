import numpy as np
import tensorflow as tf
import time
from scipy.io import FortranFile
import random

###############################################################################################
######################################### Train ###############################################

#learning iteration per same learning rate
subEpoch = 100; nStep = 5; totalEpoch = subEpoch * nStep

#initial learning rate & lr decays to lr*1/5 per N_epoch
iniLR = 0.0005

for i1 in range(nStep):
## Change Learningrate
    rate = iniLR * np.power(5.0,-float(i1))

    for i2 in range(subEpoch):

        for i3 in range(totalBatch):

            x_data, y_data = GetBatch()
            _ = sess.run(optimizer, feed_dict={x: x_data, 
                                               y: y_data,
                                               LearnRate: rate, 
                                               phase_train: True})  

###############################################################################################
###############################################################################################