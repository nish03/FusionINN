#############################################################################################
# Import Packages
#############################################################################################
import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.io import savemat

#####################Plot the images##########################################################
total_val_images = 1153

val_data_tensor = torch.load('/......./FusionINN/val_data_tensor.pt')
val_data_tensor = val_data_tensor.cpu().squeeze()
count = 0

##############################################################################################
# FusionINN 
##############################################################################################
fused_val_tensor = torch.load('/......./FusionINN/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()
#print(fused_val_tensor.shape)

recon_val_tensor = torch.load('/......./FusionINN/val_recon_tensor.pt')
recon_val_tensor = recon_val_tensor.cpu().squeeze()
#print(recon_val_tensor.shape)

ssim_fusion = torch.zeros(total_val_images)
ssim_recon  = torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_fusion_t1ce  = ssim(fused_val_tensor[i:i+1, None, :,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_fusion_flair = ssim(fused_val_tensor[i:i+1, None, :,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_fusion[i]=(ssim_fusion_t1ce + ssim_fusion_flair)/2
    ssim_recon_t1ce  = ssim(recon_val_tensor[i:i+1, 0:1, :,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_recon_flair = ssim(recon_val_tensor[i:i+1, 1:, :,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_recon[i]=(ssim_recon_t1ce + ssim_recon_flair)/2

mdic = {"ssim_fusion": ssim_fusion, "ssim_recon": ssim_recon}
savemat("/......./FusionINN/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# DDFM 
##############################################################################################
ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    fused_val_tensor = torchvision.io.read_image('/......./DDFM/output/recon/' + str(i) + '.png', torchvision.io.ImageReadMode.GRAY)/255
    ssim_t1ce  = ssim(fused_val_tensor[:, None, :,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[:, None, :,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./DDFM/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# DeepFuse 
##############################################################################################
fused_val_tensor = torch.load('/......./DeepFuse/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()

ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_t1ce  = ssim(fused_val_tensor[i:i+1, None, :,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[i:i+1, None, :,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./DeepFuse/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# FunFuseAn 
##############################################################################################
fused_val_tensor = torch.load('/......./FunFuseAn/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()

ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_t1ce  = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./FunFuseAn/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# Half-UNET 
##############################################################################################
fused_val_tensor = torch.load('/......./Half_UNET/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()

ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_t1ce  = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./Half_UNET/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# UNET 
##############################################################################################
fused_val_tensor = torch.load('/......./UNET/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()

ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_t1ce  = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./UNET/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# UNET++ 
##############################################################################################
fused_val_tensor = torch.load('/......./UNET++/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()

ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_t1ce  = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./UNET++/ssim.mat", mdic)
count += 1
print(count)

##############################################################################################
# UNET3+ 
##############################################################################################
fused_val_tensor = torch.load('/......./UNET3+/val_fused_tensor.pt')
fused_val_tensor = fused_val_tensor.cpu().squeeze()

ssim_average=torch.zeros(total_val_images)
for i in range(total_val_images):
    ssim_t1ce  = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 2:3, :, :], data_range=1)
    ssim_flair = ssim(fused_val_tensor[i:i+1, None,:,:], val_data_tensor[i:i+1, 3:,  :, :], data_range=1)
    ssim_average[i]=(ssim_t1ce + ssim_flair)/2

mdic = {"ssim": ssim_average}
savemat("/......./UNET3+/ssim.mat", mdic)
count += 1
print(count)

###############################################################################################