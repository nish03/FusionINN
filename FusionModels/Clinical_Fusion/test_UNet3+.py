import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import h5py
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.utils.tensorboard import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_train_images = 8500
total_val_images = 1153
batch_size = 64
gpu_ids = [0]

clinical_images_tensor = torch.load("/home/h1/s8993054/INN_Fusion/Clinical_Fusion/image_dataset.pt")
clinical_images_numpy  = clinical_images_tensor.cpu().squeeze()

##############################################################################################################
#UNet3+ Architecture
##############################################################################################################
""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
        
class build_unet3(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.e1=encoder_block(2,64)
    self.e2=encoder_block(64,128)
    self.e3=encoder_block(128,256)
    self.e4=encoder_block(256,512)
    
    self.b=conv_block(512,1024)
    
    self.CatChannels=64
    self.CatBlocks=5
    self.UpChannels=self.CatChannels*self.CatBlocks
    
    self.h1_PT_hd4=nn.MaxPool2d(8,8,ceil_mode=True)
    self.h1_PT_hd4_conv=nn.Conv2d(64,64,kernel_size=3,padding=1)
    self.h1_PT_hd4_bn=nn.BatchNorm2d(self.CatChannels)
    self.h1_PT_hd4_relu=nn.ReLU(inplace=True)
    
    self.h2_PT_hd4=nn.MaxPool2d(4,4,ceil_mode=True)
    self.h2_PT_hd4_conv=nn.Conv2d(128,64,kernel_size=3,padding=1)
    self.h2_PT_hd4_bn=nn.BatchNorm2d(self.CatChannels)
    self.h2_PT_hd4_relu=nn.ReLU(inplace=True)
    
    self.h3_PT_hd4=nn.MaxPool2d(2,2,ceil_mode=True)
    self.h3_PT_hd4_conv=nn.Conv2d(256,64,3,padding=1)
    self.h3_PT_hd4_bn=nn.BatchNorm2d(self.CatChannels)
    self.h3_PT_hd4_relu=nn.ReLU(inplace=True)
    
    self.h4_Cat_hd4_conv=nn.Conv2d(512,64,3,padding=1)
    self.h4_Cat_hd4_bn=nn.BatchNorm2d(self.CatChannels)
    self.h4_Cat_hd4_relu=nn.ReLU(inplace=True)
    
    self.hd5_UT_hd4=nn.Upsample(scale_factor=2,mode='bilinear')
    self.hd5_UT_hd4_conv=nn.Conv2d(1024,64,3,padding=1)
    self.hd5_UT_hd4_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd5_UT_hd4_relu=nn.ReLU(inplace=True)
    
    self.conv4d_1=nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
    self.bn4d_1=nn.BatchNorm2d(self.UpChannels)
    self.relu4d_1=nn.ReLU(inplace=True)
    
    
    self.h1_PT_hd3=nn.MaxPool2d(4,4,ceil_mode=True)
    self.h1_PT_hd3_conv=nn.Conv2d(64,64,kernel_size=3,padding=1)
    self.h1_PT_hd3_bn=nn.BatchNorm2d(self.CatChannels)
    self.h1_PT_hd3_relu=nn.ReLU(inplace=True)
    
    self.h2_PT_hd3=nn.MaxPool2d(2,2,ceil_mode=True)
    self.h2_PT_hd3_conv=nn.Conv2d(128,64,kernel_size=3,padding=1)
    self.h2_PT_hd3_bn=nn.BatchNorm2d(self.CatChannels)
    self.h2_PT_hd3_relu=nn.ReLU(inplace=True)
    
    self.h3_Cat_hd3_conv=nn.Conv2d(256,64,3,padding=1)
    self.h3_Cat_hd3_bn=nn.BatchNorm2d(self.CatChannels)
    self.h3_Cat_hd3_relu=nn.ReLU(inplace=True)
    
    self.hd4_UT_hd3=nn.Upsample(scale_factor=2,mode='bilinear')
    self.hd4_UT_hd3_conv=nn.Conv2d(self.UpChannels,64,3,padding=1)
    self.hd4_UT_hd3_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd4_UT_hd3_relu=nn.ReLU(inplace=True)
    
    self.hd5_UT_hd3=nn.Upsample(scale_factor=4,mode='bilinear')
    self.hd5_UT_hd3_conv=nn.Conv2d(1024,64,3,padding=1)
    self.hd5_UT_hd3_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd5_UT_hd3_relu=nn.ReLU(inplace=True)
    
    self.conv3d_1=nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
    self.bn3d_1=nn.BatchNorm2d(self.UpChannels)
    self.relu3d_1=nn.ReLU(inplace=True)
    
    
    self.h1_PT_hd2=nn.MaxPool2d(2,2,ceil_mode=True)
    self.h1_PT_hd2_conv=nn.Conv2d(64,64,3,padding=1)
    self.h1_PT_hd2_bn=nn.BatchNorm2d(self.CatChannels)
    self.h1_PT_hd2_relu=nn.ReLU(inplace=True)
    
    self.h2_Cat_hd2_conv=nn.Conv2d(128,64,3,padding=1)
    self.h2_Cat_hd2_bn=nn.BatchNorm2d(self.CatChannels)
    self.h2_Cat_hd2_relu=nn.ReLU(inplace=True)
    
    self.hd3_UT_hd2=nn.Upsample(scale_factor=2,mode='bilinear')
    self.hd3_UT_hd2_conv=nn.Conv2d(self.UpChannels,64,3,padding=1)
    self.hd3_UT_hd2_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd3_UT_hd2_relu=nn.ReLU(inplace=True)
    
    self.hd4_UT_hd2=nn.Upsample(scale_factor=4,mode='bilinear')
    self.hd4_UT_hd2_conv=nn.Conv2d(self.UpChannels,64,3,padding=1)
    self.hd4_UT_hd2_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd4_UT_hd2_relu=nn.ReLU(inplace=True)
    
    self.hd5_UT_hd2=nn.Upsample(scale_factor=8,mode='bilinear')
    self.hd5_UT_hd2_conv=nn.Conv2d(1024,64,3,padding=1)
    self.hd5_UT_hd2_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd5_UT_hd2_relu=nn.ReLU(inplace=True)
    
    self.conv2d_1=nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
    self.bn2d_1=nn.BatchNorm2d(self.UpChannels)
    self.relu2d_1=nn.ReLU(inplace=True)
    
    
    self.h1_Cat_hd1_conv=nn.Conv2d(64,64,3,padding=1)
    self.h1_Cat_hd1_bn=nn.BatchNorm2d(self.CatChannels)
    self.h1_Cat_hd1_relu=nn.ReLU(inplace=True)
    
    self.hd2_UT_hd1=nn.Upsample(scale_factor=2,mode='bilinear')
    self.hd2_UT_hd1_conv=nn.Conv2d(self.UpChannels,64,3,padding=1)
    self.hd2_UT_hd1_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd2_UT_hd1_relu=nn.ReLU(inplace=True)
    
    self.hd3_UT_hd1=nn.Upsample(scale_factor=4,mode='bilinear')
    self.hd3_UT_hd1_conv=nn.Conv2d(self.UpChannels,64,3,padding=1)
    self.hd3_UT_hd1_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd3_UT_hd1_relu=nn.ReLU(inplace=True)
    
    self.hd4_UT_hd1=nn.Upsample(scale_factor=8,mode='bilinear')
    self.hd4_UT_hd1_conv=nn.Conv2d(self.UpChannels,64,3,padding=1)
    self.hd4_UT_hd1_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd4_UT_hd1_relu=nn.ReLU(inplace=True)
    
    self.hd5_UT_hd1=nn.Upsample(scale_factor=16,mode='bilinear')
    self.hd5_UT_hd1_conv=nn.Conv2d(1024,64,3,padding=1)
    self.hd5_UT_hd1_bn=nn.BatchNorm2d(self.CatChannels)
    self.hd5_UT_hd1_relu=nn.ReLU(inplace=True)
    
    self.conv1d_1=nn.Conv2d(self.UpChannels,self.UpChannels,3,padding=1)
    self.bn1d_1=nn.BatchNorm2d(self.UpChannels)
    self.relu1d_1=nn.ReLU(inplace=True)
    
    self.outconv1=nn.Conv2d(self.UpChannels,1,3,padding=1)
  
  def forward(self,inputs):
    ## -----------Encoder----------
    h1,h2=self.e1(inputs)
    h2,h3=self.e2(h2)
    h3,h4=self.e3(h3)
    h4,h5=self.e4(h4)
    
    hd5=self.b(h5)
    
    ## -----------Decoder-----------
    
    
    h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
    h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
    h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
    h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
    hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
    hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels
 

    h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
    h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
    h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
    hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
    hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
    hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

    h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
    h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
    hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
    hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
    hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
    hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

    h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
    hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
    hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
    hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
    hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
    hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

    d1 = self.outconv1(hd1)
    out=nn.Sigmoid()(d1)
    
    return out
    
    
model= build_unet3().to(device)
model = model.float()
model = torch.nn.DataParallel(model, gpu_ids)
###############################################################################################################
# Load the data and the model
###############################################################################################################
clinical_images_tensor = clinical_images_tensor.float()

# load pretrain model
print('using pretrained model')
state_dict = torch.load('/home/h1/s8993054/INN_Fusion/UNET3+/unet3+.pt',map_location='cuda')['model_state_dict']
model.load_state_dict(state_dict)
model.to(device)
model = model.float()
model = torch.nn.DataParallel(model, gpu_ids)
###############################################################################################################
# start the testing loop
###############################################################################################################
model.eval()
fused_val = torch.zeros(clinical_images_tensor.shape[0], 1, 240,240)
count = 0
with torch.no_grad():
    for idx in range(clinical_images_tensor.shape[0]):
        test_val = clinical_images_tensor[idx:idx+1,:,:,:].to(device)
        fused_test = model(test_val)
        fused_val[idx,] = fused_test[0,]
        count += 1
        if count % 100 == 0:
            print(count)   
    torch.save(fused_val, 'clinical_fused_tensor_unet3+.pt')

clinical_fused_numpy = fused_val.cpu().squeeze()
#save the clinical images
for i in range(clinical_images_numpy.shape[0]):
    plt.imsave('/home/h1/s8993054/INN_Fusion/Clinical_Fusion/Fused_UNet3+/fused_' + str(i) + '.png', clinical_fused_numpy[i,], cmap = 'gray')
    #plt.clf()
    plt.close()
        