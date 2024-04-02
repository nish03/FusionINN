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
#FunFuseAn
class FunFuseAn(nn.Module):
    def  __init__(self):
        super(FunFuseAn, self).__init__()
        #####mri lf layer 1#####
        self.mri_lf = nn.Sequential( #input shape (,1,240,240)
                         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=4),
                         nn.BatchNorm2d(16),
                         nn.LeakyReLU(0.2,inplace=True)) #output shape (,16,240,240)   
        #####mri hf layers#####
        self.mri_hf = nn.Sequential(  #input shape (,1,256,256)
                         nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size  = 3, stride= 1, padding = 1),
                         nn.BatchNorm2d(16),
                         nn.LeakyReLU(0.2,inplace=True),
                         nn.Conv2d(in_channels  = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                         nn.BatchNorm2d(32),
                         nn.LeakyReLU(0.2,inplace=True),
                         nn.Conv2d(in_channels  = 32, out_channels = 64, kernel_size  = 3, stride = 1, padding = 1),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(0.2,inplace=True)) #output shape (,64,240,240)
        #####pet lf layer 1#####
        self.pet_lf = nn.Sequential( #input shape (,1,256,256)
                         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
                         nn.BatchNorm2d(16),
                         nn.LeakyReLU(0.2,inplace=True)) #output shape (,16,240,240)   
        #####pet hf layers#####
        self.pet_hf = nn.Sequential(  #input shape (,1,256,256)
                         nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size  = 5, stride= 1, padding = 2),
                         nn.BatchNorm2d(16),
                         nn.LeakyReLU(0.2,inplace=True),
                         nn.Conv2d(in_channels  = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2),
                         nn.BatchNorm2d(32),
                         nn.LeakyReLU(0.2,inplace=True),
                         nn.Conv2d(in_channels  = 32, out_channels = 64, kernel_size  = 3, stride = 1, padding = 1),
                         nn.BatchNorm2d(64),
                         nn.LeakyReLU(0.2,inplace=True)) #output shape (,64,240,240)
        #####reconstruction layer 1#####
        self.recon1 = nn.Sequential(  #input shape (, 64, 256, 256)
                          nn.Conv2d(in_channels  = 64,  out_channels = 32, kernel_size  = 5, stride = 1, padding = 2),
                          nn.BatchNorm2d(32),
                          nn.LeakyReLU(0.2,inplace=True),
                          nn.Conv2d(in_channels  = 32, out_channels = 16, kernel_size  = 5, stride = 1, padding = 2),
                          nn.BatchNorm2d(16),
                          nn.LeakyReLU(0.2,inplace=True)) #output shape (,16, 240, 240)
        
        #####reconstruction layer 2#####
        self.recon2 = nn.Sequential(      #input shape (,16, 240, 240)
                            nn.Conv2d(in_channels  = 16, out_channels = 1, kernel_size  = 5, stride = 1, padding = 2))   #output shape (,1,240,240)

    def forward(self, x, y):
        #mri lf layer 1
        x1 = self.mri_lf(x)
        #mri hf layers
        x2 = self.mri_hf(x)
        #pet lf layer 1
        y1 = self.pet_lf(y)
        #pet hf layers
        y2 = self.pet_hf(y)
        #high frequency fusion layer, add epsilon: 1e-8 to avoid division by zero during training
        fuse_hf = torch.maximum(x2,y2) / (x2 + y2 + 1e-8)
        #reconstruction layer1
        recon_hf = self.recon1(fuse_hf)
        #low frequency fusion layer
        fuse_lf = (x1 + y1 + recon_hf)/3
        #reconstruction layer2
        recon3 = self.recon2(fuse_lf)
        #tanh layer
        #fused = torch.tanh(recon3)    
        return torch.sigmoid(recon3)
        
model= FunFuseAn().to(device)
model = model.float()
model=torch.nn.DataParallel(model, gpu_ids)   

##############################################################################################################
def Load_Data():
    # loading train data
    hf = h5py.File("/......../train_data.h5", 'r')
    train_data = hf['data'][()]  # `data` is now an ndarray
    hf.close()

    # loading val data
    hf = h5py.File("/......../test_data.h5", 'r')
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
if os.path.exists('FunFuseAn.pt'):
    print('using pretrained model')
    state_dict = torch.load('FunFuseAn.pt',map_location='cuda')['model_state_dict']
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
        fused_test = model(test_val[:,0:1,:,:], test_val[:,1:,:,:])
        fused_val[idx,] = fused_test[0,]
        count += 1
        if count % 100 == 0:
            print(count)
    torch.save(fused_val, 'val_fused_tensor.pt')
        