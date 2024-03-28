import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.io import savemat

#####################Plot the images##########################################################
total_val_images = 1153

val_data_tensor = torch.load('/home/h1/s8993054/INN_Fusion/INN/val_data_tensor.pt')
#val_data_tensor = val_data_tensor.cpu().squeeze()
count = 0

##############################################################################################
# Fusion SSIM  
##############################################################################################
fused_val_tensor = torch.load('/home/h1/s8993054/INN_Fusion/INN/val_fused_tensor_400.pt')
#fused_val_tensor = fused_val_tensor.cpu().squeeze()
print(fused_val_tensor.shape)

recon_val_tensor = torch.load('/home/h1/s8993054/INN_Fusion/INN/val_recon_tensor_400.pt')
#recon_val_tensor = recon_val_tensor.cpu().squeeze()
print(recon_val_tensor.shape)

ssim_fusion_t1ce=torch.zeros(total_val_images)
ssim_fusion_flair=torch.zeros(total_val_images)
ssim_recon_t1ce=torch.zeros(total_val_images)
ssim_recon_flair=torch.zeros(total_val_images)

for i in range(total_val_images):
    ssim_fusion_t1ce[i]  = ssim(fused_val_tensor[i:i+1, :, :,:].cuda(), val_data_tensor[i:i+1, 2:3, :, :].cuda(), data_range=1).cuda()
    ssim_fusion_flair[i] = ssim(fused_val_tensor[i:i+1, :, :,:].cuda(), val_data_tensor[i:i+1, 3:,  :, :].cuda(), data_range=1).cuda()
    ssim_recon_t1ce[i]  = ssim(recon_val_tensor[i:i+1, 0:1, :,:].cuda(), val_data_tensor[i:i+1, 2:3, :, :].cuda(), data_range=1).cuda()
    ssim_recon_flair[i] = ssim(recon_val_tensor[i:i+1, 1:, :,:].cuda(), val_data_tensor[i:i+1, 3:,  :, :].cuda(), data_range=1).cuda()
    #ssim_average[i]=(ssim_t1ce + ssim_flair)/2

#torch.save(ssim_average, '/home/h1/s8993054/INN_Fusion/INN_2_layer_0.8/ssim_scores.pt')
mdic = {"ssim_fusion_t1ce": ssim_fusion_t1ce, "ssim_fusion_flair": ssim_fusion_flair, "ssim_recon_t1ce": ssim_recon_t1ce, "ssim_recon_flair": ssim_recon_flair }
savemat("/home/h1/s8993054/INN_Fusion/INN/ssim_detailed_400.mat", mdic)
count += 1
print(count)


ssim_fusion_t1ce_mean = torch.mean(ssim_fusion_t1ce)
ssim_fusion_flair_mean = torch.mean(ssim_fusion_flair)
ssim_recon_t1ce_mean = torch.mean(ssim_recon_t1ce)
ssim_recon_flair_mean = torch.mean(ssim_recon_flair)

print('SSIM Fusion T1ce:', ssim_fusion_t1ce_mean)
print('SSIM Fusion Flair:', ssim_fusion_flair_mean)
print('SSIM Recon T1ce:', ssim_recon_t1ce_mean)
print('SSIM Recon Flair:', ssim_recon_flair_mean)