import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.io import savemat


################################################################
#DeepFuse
################################################################
Q_INN    = sio.loadmat('/home/h1/s8993054/INN_Fusion/DDFM/Q.mat')['Q']
ssim_INN = sio.loadmat('/home/h1/s8993054/INN_Fusion/DDFM/ssim.mat')['ssim']

Q_INN = np.mean(Q_INN, axis = 0)
ssim_INN = np.mean(ssim_INN) 

print('DDFM Feature Mutual Information (FMI):', Q_INN[0])
print('DDFM Nonlinear Correlation Information Entropy (NCIE):', Q_INN[1])
print('DDFM Xydeas:', Q_INN[2])
print('DDFM Peilla:', Q_INN[3])
print('DDFM Visual Information Fidelity (VIFF):', Q_INN[4])
print('DDFM Structural Similarity Index (SSIM):', ssim_INN)


