#############################################################################################
# Import Packages
#############################################################################################
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.io import savemat

#############################################################################################
#FusionINN
#############################################################################################
Q_INN    = sio.loadmat('/home/h1/s8993054/INN_Fusion/INN/Q.mat')['Q']
ssim_INN = sio.loadmat('/home/h1/s8993054/INN_Fusion/INN/ssim.mat')['ssim']

Q_INN = np.mean(Q_INN, axis = 0)
ssim_INN = np.mean(ssim_INN) 

print('INN Feature Mutual Information (FMI):', Q_INN[0])
print('INN Nonlinear Correlation Information Entropy (NCIE):', Q_INN[1])
print('INN Xydeas:', Q_INN[2])
print('INN Peilla:', Q_INN[3])
print('INN Structural Similarity Index (SSIM):', ssim_INN)

#############################################################################################
#DDFM
#############################################################################################
Q_DDFM    = sio.loadmat('/home/h1/s8993054/INN_Fusion/DDFM/Q.mat')['Q']
ssim_DDFM = sio.loadmat('/home/h1/s8993054/INN_Fusion/DDFM/ssim.mat')['ssim']

Q_DDFM = np.mean(Q_DDFM, axis = 0)
ssim_DDFM = np.mean(ssim_DDFM) 

print('DDFM Feature Mutual Information (FMI):', Q_DDFM[0])
print('DDFM Nonlinear Correlation Information Entropy (NCIE):', Q_DDFM[1])
print('DDFM Xydeas:', Q_DDFM[2])
print('DDFM Peilla:', Q_DDFM[3])
print('DDFM Structural Similarity Index (SSIM):', ssim_DDFM)

#############################################################################################
#DeepFuse
#############################################################################################
Q_DeepFuse    = sio.loadmat('/home/h1/s8993054/INN_Fusion/DeepFuse/Q.mat')['Q']
ssim_DeepFuse = sio.loadmat('/home/h1/s8993054/INN_Fusion/DeepFuse/ssim.mat')['ssim']

Q_DeepFuse = np.mean(Q_DeepFuse, axis = 0)
ssim_DeepFuse = np.mean(ssim_DeepFuse) 

print('DeepFuse Feature Mutual Information (FMI):', Q_DeepFuse[0])
print('DeepFuse Nonlinear Correlation Information Entropy (NCIE):', Q_DeepFuse[1])
print('DeepFuse Xydeas:', Q_DeepFuse[2])
print('DeepFuse Peilla:', Q_DeepFuse[3])
print('DeepFuse Structural Similarity Index (SSIM):', ssim_DeepFuse)

#############################################################################################
#FunFuseAn
#############################################################################################
Q_FunFuseAn    = sio.loadmat('/home/h1/s8993054/INN_Fusion/FunFuseAn/Q.mat')['Q']
ssim_FunFuseAn = sio.loadmat('/home/h1/s8993054/INN_Fusion/FunFuseAn/ssim.mat')['ssim']

Q_FunFuseAn = np.mean(Q_FunFuseAn, axis = 0)
ssim_FunFuseAn = np.mean(ssim_FunFuseAn) 

print('FunFuseAn Feature Mutual Information (FMI):', Q_FunFuseAn[0])
print('FunFuseAn Nonlinear Correlation Information Entropy (NCIE):', Q_FunFuseAn[1])
print('FunFuseAn Xydeas:', Q_FunFuseAn[2])
print('FunFuseAn Peilla:', Q_FunFuseAn[3])
print('FunFuseAn Structural Similarity Index (SSIM):', ssim_FunFuseAn)

############################################################################################
#Half_UNET
############################################################################################
Q_Half_UNET    = sio.loadmat('/home/h1/s8993054/INN_Fusion/Half_UNET/Q.mat')['Q']
ssim_Half_UNET = sio.loadmat('/home/h1/s8993054/INN_Fusion/Half_UNET/ssim.mat')['ssim']

Q_Half_UNET = np.mean(Q_Half_UNET, axis = 0)
ssim_Half_UNET = np.mean(ssim_Half_UNET) 

print('Half_UNET Feature Mutual Information (FMI):', Q_Half_UNET[0])
print('Half_UNET Nonlinear Correlation Information Entropy (NCIE):', Q_Half_UNET[1])
print('Half_UNET Xydeas:', Q_Half_UNET[2])
print('Half_UNET Peilla:', Q_Half_UNET[3])
print('Half_UNET Structural Similarity Index (SSIM):', ssim_Half_UNET)

############################################################################################
#UNET
############################################################################################
Q_UNET    = sio.loadmat('/home/h1/s8993054/INN_Fusion/UNET/Q.mat')['Q']
ssim_UNET = sio.loadmat('/home/h1/s8993054/INN_Fusion/UNET/ssim.mat')['ssim']

Q_UNET = np.mean(Q_UNET, axis = 0)
ssim_UNET = np.mean(ssim_UNET) 

print('UNET Feature Mutual Information (FMI):', Q_UNET[0])
print('UNET Nonlinear Correlation Information Entropy (NCIE):', Q_UNET[1])
print('UNET Xydeas:', Q_UNET[2])
print('UNET Peilla:', Q_UNET[3])
print('UNET Structural Similarity Index (SSIM):', ssim_UNET)

############################################################################################
#UNET++
############################################################################################
Q_UNET_plus_plus    = sio.loadmat('/home/h1/s8993054/INN_Fusion/UNET++/Q.mat')['Q']
ssim_UNET_plus_plus = sio.loadmat('/home/h1/s8993054/INN_Fusion/UNET++/ssim.mat')['ssim']

Q_UNET_plus_plus = np.mean(Q_UNET_plus_plus, axis = 0)
ssim_UNET_plus_plus = np.mean(ssim_UNET_plus_plus) 

print('UNET++ Feature Mutual Information (FMI):', Q_UNET_plus_plus[0])
print('UNET++ Nonlinear Correlation Information Entropy (NCIE):', Q_UNET_plus_plus[1])
print('UNET++ Xydeas:', Q_UNET_plus_plus[2])
print('UNET++ Peilla:', Q_UNET_plus_plus[3])
print('UNET++ Structural Similarity Index (SSIM):', ssim_UNET_plus_plus)

###########################################################################################
#UNET3+
###########################################################################################
Q_UNET_three_plus    = sio.loadmat('/home/h1/s8993054/INN_Fusion/UNET3+/Q.mat')['Q']
ssim_UNET_three_plus = sio.loadmat('/home/h1/s8993054/INN_Fusion/UNET3+/ssim.mat')['ssim']

Q_UNET_three_plus = np.mean(Q_UNET_three_plus, axis = 0)
ssim_UNET_three_plus = np.mean(ssim_UNET_three_plus) 

print('UNET3+ Feature Mutual Information (FMI):', Q_UNET_three_plus[0])
print('UNET3+ Nonlinear Correlation Information Entropy (NCIE):', Q_UNET_three_plus[1])
print('UNET3+ Xydeas:', Q_UNET_three_plus[2])
print('UNET3+ Peilla:', Q_UNET_three_plus[3])
print('UNET3+ Structural Similarity Index (SSIM):', ssim_UNET_three_plus)


###########################################################################################