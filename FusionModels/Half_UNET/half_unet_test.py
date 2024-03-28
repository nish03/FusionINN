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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_train_images = 8500
total_val_images = 1153
batch_size = 64
gpu_ids = [0]


##################################################################################################################
#Half-Unet Architecture
#Paper title: Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation
class SeparableConv2d(nn.Module):
  def __init__(self,in_c,out_c):
    super(SeparableConv2d,self).__init__()
    self.depthwise=nn.Conv2d(in_c,in_c,kernel_size=3,groups=in_c,padding=1)
    self.pointwise=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
    
  def forward(self,x):
    out=self.depthwise(x)
    out=self.pointwise(x)
    return out


class ghost_block(nn.Module):
  def __init__(self,in_c,out_c):
    super().__init__()
    
    self.conv1=nn.Conv2d(in_c,out_c,kernel_size=3,padding=1)
    self.bn1=nn.BatchNorm2d(out_c)
    self.relu=nn.ReLU()
    self.conv2=SeparableConv2d(out_c,out_c)
  def forward(self,x):
    out=self.conv1(x)
    out=self.bn1(out)
    out=self.relu(out)
    out=self.conv2(out)
    return out
    
class unet_encoder(nn.Module):
  def __init__(self,in_c,out_c):
    super().__init__()
    self.conv=ghost_block(in_c,out_c)
    self.pool=nn.MaxPool2d((2,2))
    
  def forward(self,inputs):
    x=self.conv(inputs)
    p=self.pool(x)
    return x,p

class unet_decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.up=nn.UpsamplingBilinear2d(scale_factor=2)
    
  def forward(self,inputs):
    
    x=self.up(inputs)
    
    return x
    
class Half_unet(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.e1=unet_encoder(2,64)
    self.e2=unet_encoder(64,64)
    self.e3=unet_encoder(64,64)
    self.e4=unet_encoder(64,64)
    
    self.b1=ghost_block(64,64)
    
    self.d1=unet_decoder()
    self.d2=unet_decoder()
    self.d3=unet_decoder()
    self.d4=unet_decoder()
    
    self.output=nn.Conv2d(320,1,kernel_size=1,padding=0)
  
  def forward(self,inputs):
    s1,p1=self.e1(inputs)
    s2,p2=self.e2(p1)
    s3,p3=self.e3(p2)
    s4,p4=self.e4(p3)
    
    b= self.b1(p4)
    
    d1=self.d1(b)
    d1=torch.cat([d1,s4],axis=1)
    d2=self.d2(d1)
    d2=torch.cat([d2,s3],axis=1)
    d3=self.d3(d2)
    d3=torch.cat([d3,s2],axis=1)
    d4=self.d4(d3)
    d4=torch.cat([d4,s1],axis=1)
   
    out=self.output(d4)
    output=nn.Sigmoid()(out)
    return output
    
model= Half_unet().to(device)
model = model.float()
model=torch.nn.DataParallel(model,gpu_ids)  

##############################################################################################################
def Load_Data():
    # loading train data
    hf = h5py.File("/projects/p084/p_discoret/Brats2018_training_data_sep_channels_train_val_mix.h5", 'r')
    train_data = hf['data'][()]  # `data` is now an ndarray
    hf.close()

    # loading val data
    hf = h5py.File("/projects/p084/p_discoret/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
    val_data = hf['data'][()]  # `data` is now an ndarray
    hf.close()

    print("Training and Validation data has been loaded")
    return train_data, val_data


def Normalize_Data(train_data, val_data):
    for i in range(len(train_data)):
        for j in range(4):
            train_data[i, j, :, :] = (train_data[i, j, :, :] - np.min(train_data[i, j, :, :])) / (
                    np.max(train_data[i, j, :, :]) - np.min(train_data[i, j, :, :]))

    for i in range(len(val_data)):
        for j in range(4):
            val_data[i, j, :, :] = (val_data[i, j, :, :] - np.min(val_data[i, j, :, :])) / (
                    np.max(val_data[i, j, :, :]) - np.min(val_data[i, j, :, :]))

    train_data_tensor = torch.from_numpy(train_data).float()
    val_data_tensor = torch.from_numpy(val_data).float()
    return train_data_tensor, val_data_tensor


###############################################################################################################
# Load the data and the model
###############################################################################################################
train_data, val_data = Load_Data()
train_data_tensor, val_data_tensor = Normalize_Data(train_data, val_data)

# load pretrain model
if os.path.exists('half_unet.pt'):
    print('using pretrained model')
    state_dict = torch.load('half_unet.pt',map_location='cuda')['model_state_dict']
    model.load_state_dict(state_dict)
model.to(device)
model = model.float()
model = torch.nn.DataParallel(model, gpu_ids)

###############################################################################################################
# start the testing loop
###############################################################################################################
model.eval()
#val_batch_idxs=total_val_images//batch_size
fused_val = torch.zeros(total_val_images, 1, 240,240)
count = 0
with torch.no_grad():
    for idx in range(total_val_images):
        test_val = val_data_tensor[idx:idx+1,2:,:,:].to(device)
        fused_test = model(test_val)
        fused_val[idx,] = fused_test[0,]
        count += 1
        if count % 100 == 0:
            print(count)
    torch.save(fused_val, 'val_fused_tensor.pt')
        