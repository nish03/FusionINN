import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.io import savemat
import torchvision.io

#####################Plot the images##########################################################
total_val_images = 1153

val_data_tensor = torch.load('/home/h1/s8993054/INN_Fusion/INN/val_data_tensor.pt')
val_data_tensor = val_data_tensor.cpu().squeeze()
count = 0

##############################################################################################
# DDFM 
##############################################################################################
ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    fused_val_tensor = torchvision.io.read_image('/home/h1/s8993054/INN_Fusion/DDFM/output/recon/' + str(i) + '.png', torchvision.io.ImageReadMode.GRAY)/255
    ssim_t1ce  = ssim(fused_val_tensor[:, None, :,:].cuda(), val_data_tensor[i:i+1, 2:3, :, :].cuda(), data_range=1).cuda()
    ssim_flair = ssim(fused_val_tensor[:, None, :,:].cuda(), val_data_tensor[i:i+1, 3:,  :, :].cuda(), data_range=1).cuda()
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

torch.save(ssim_average, '/home/h1/s8993054/INN_Fusion/DDFM/ssim_scores.pt')
mdic = {"ssim": ssim_average}
savemat("/home/h1/s8993054/INN_Fusion/DDFM/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################