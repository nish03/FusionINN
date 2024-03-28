import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import h5py
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
import os


############################################Plot the images###########################################################
total_val_images = 1153

fused_images = torch.load('val_fused_tensor.pt')
fused_images = fused_images.cpu().squeeze()

input_images = torch.load('/home/h1/s8993054/INN_Fusion/INN/val_data_tensor.pt')
input_images = input_images.cpu().squeeze()


for i in range(total_val_images):
    plt.imsave('/home/h1/s8993054/INN_Fusion/Half_UNET/Fused_normalized/fused_' + str(i) + '.png', fused_images[i,], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.close()
       
############################################Plot the Loss curves#######################################################
ep_ssim_train_loss_t1ce = np.array(torch.load('half_unet.pt',map_location='cpu')['training_loss_ssim_t1ce'])
ep_ssim_val_loss_t1ce   = np.array(torch.load('half_unet.pt',map_location='cpu')['validation_loss_ssim_t1ce'])

ep_ssim_train_loss_flair = np.array(torch.load('half_unet.pt',map_location='cpu')['training_loss_ssim_flair'])
ep_ssim_val_loss_flair   = np.array(torch.load('half_unet.pt',map_location='cpu')['validation_loss_ssim_flair'])

plt.plot(ep_ssim_train_loss_t1ce, label = 'Fusion - ssim-train-t1ce')
plt.plot(ep_ssim_val_loss_t1ce, label = 'Fusion - ssim-val-t1ce')
plt.plot(ep_ssim_train_loss_flair, label = 'Fusion - ssim-train-flair')
plt.plot(ep_ssim_val_loss_flair, label = 'Fusion - ssim-val-flair')
plt.legend()
plt.savefig('/home/h1/s8993054/INN_Fusion/Half_UNET/loss_fusion_ssim.png', dpi = 200)
plt.close()









