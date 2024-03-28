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


fused_images = torch.load('val_fused_tensor_400.pt')
fused_images = fused_images.cpu().squeeze()

recon_images = torch.load('val_recon_tensor_400.pt')
recon_images = recon_images.cpu().squeeze()

input_images = torch.load('val_data_tensor.pt')
input_images = input_images.cpu().squeeze()


for i in range(total_val_images):
    plt.imsave('/home/h1/s8993054/INN_Fusion/INN/Fused_normalized/fused_' + str(i) + '.png', fused_images[i,], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.clf()
    plt.imsave('/home/h1/s8993054/INN_Fusion/INN/Recon_T1ce_normalized/Recon_T1ce_' + str(i) + '.png', recon_images[i,0,], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.clf()
    plt.imsave('/home/h1/s8993054/INN_Fusion/INN/Recon_Flair_normalized/Recon_Flair_' + str(i) + '.png', recon_images[i,1,], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.close()
    #plt.imsave('/home/h1/s8993054/INN_Fusion/INN/T1ce/T1ce_' + str(i) + '.png', input_images[i,2,], cmap = 'gray')
    #plt.clf()
    #plt.imsave('/home/h1/s8993054/INN_Fusion/INN/Flair/Flair_' + str(i) + '.png', input_images[i,3,], cmap = 'gray')
    #plt.close()
    
       
############################################Plot the Loss curves#######################################################
ep_ssim_train_loss_t1ce = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['training_loss_ssim_t1ce'])
ep_ssim_val_loss_t1ce   = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['validation_loss_ssim_t1ce'])

ep_ssim_train_loss_flair = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['training_loss_ssim_flair'])
ep_ssim_val_loss_flair   = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['validation_loss_ssim_flair'])

ep_ssim_rev_train_loss_t1ce = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['training_loss_ssim_rev_t1ce'])
ep_ssim_rev_val_loss_t1ce   = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['validation_loss_ssim_rev_t1ce'])

ep_ssim_rev_train_loss_flair = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['training_loss_ssim_rev_flair'])
ep_ssim_rev_val_loss_flair   = np.array(torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cpu')['validation_loss_ssim_rev_flair'])

plt.plot(ep_ssim_train_loss_t1ce, label = 'Fusion - ssim-train-t1ce')
plt.plot(ep_ssim_val_loss_t1ce, label = 'Fusion - ssim-val-t1ce')
plt.plot(ep_ssim_train_loss_flair, label = 'Fusion - ssim-train-flair')
plt.plot(ep_ssim_val_loss_flair, label = 'Fusion - ssim-val-flair')
plt.legend()
plt.savefig('/home/h1/s8993054/INN_Fusion/INN/Loss_curves/loss_fusion_ssim.png', dpi = 200)
plt.close()


plt.plot(ep_ssim_rev_train_loss_t1ce, label = 'Recon - ssim-train-t1ce')
plt.plot(ep_ssim_rev_val_loss_t1ce, label = 'Recon - ssim-val-t1ce')
plt.plot(ep_ssim_rev_train_loss_flair, label = 'Recon - ssim-train-flair')
plt.plot(ep_ssim_rev_val_loss_flair, label = 'Recon - ssim-val-flair')
plt.legend()
plt.savefig('/home/h1/s8993054/INN_Fusion/INN/Loss_curves/loss_recon_ssim.png', dpi = 200)
plt.close()








