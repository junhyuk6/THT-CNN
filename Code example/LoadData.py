import numpy as np
import tensorflow as tf
import time
from scipy.io import FortranFile
import random

######################################################################################################
########################################## Loaddata ##################################################

##Data Parameters
#grid size of data field
nxp = 192; nzp = 192;

def LoadData(nField,iniField,intervalField):
    #r.m.s of data (0 : du/dy, 1 : dw/dy, 2 : p)
    Input_std = np.empty([3])
    Input_std[0] = 66.274826292; Input_std[1] = 37.0097954241; Input_std[2] = 1.58318422533;
    Data = np.empty([nField,nzp,nxp,4]) # [number of field, z-grids, x-grids, variables(du/dy, dw/dy, p, dT/dy)]
    for j in range(nField) :
        p = iniField + j*intervalField
        Filename = '/DLdata/' + '%05d'%(p)

        f = FortranFile(Filename,'r') 
        insfield = f.read_reals(dtype='float32')
        Data[j,:,:,:] = np.transpose(np.reshape(insfield,(4,nxp,nzp), order='F'))

    for image1 in range(3):
        Data[:,:,:,image1] = Data[:,:,:,image1]/Input_std[image1]
    return Data

nTrainField = 100
TrainData = LoadTrainData(nTrainField,0,1)
