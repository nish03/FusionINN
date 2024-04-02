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
#DeepFuse
class ConvLayer_DeepFuse(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 16, kernel_size = 5, last = nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class FusionLayer(nn.Module):
    def forward(self, x, y):
        return x + y


class DeepFuse(nn.Module):
    def __init__(self):
        super(DeepFuse, self).__init__()
        self.layer1 = ConvLayer_DeepFuse(1, 16, 5, last = nn.LeakyReLU)
        self.layer2 = ConvLayer_DeepFuse(16, 32, 7)
        self.layer3 = FusionLayer()
        self.layer4 = ConvLayer_DeepFuse(32, 32, 7, last = nn.LeakyReLU)
        self.layer5 = ConvLayer_DeepFuse(32, 16, 5, last = nn.LeakyReLU)
        self.layer6 = ConvLayer_DeepFuse(16, 1, 5, last = nn.Tanh)

    

    def forward(self,y_1,y_2):
        c11 = self.layer1(y_1)
        c12 = self.layer1(y_2)
        c21 = self.layer2(c11)
        c22 = self.layer2(c12)
        f_m = self.layer3(c21, c22)
        c3  = self.layer4(f_m)
        c4  = self.layer5(c3)
        c5  = self.layer6(c4)
        return c5
 
model= DeepFuse().to(device)
model = model.float()
model=torch.nn.DataParallel(model, gpu_ids)   

##############################################################################################################
def Load_Data():
    # loading train data
    hf = h5py.File("/....../train_data.h5", 'r')
    train_data = hf['data'][()]  # `data` is now an ndarray
    hf.close()

    # loading val data
    hf = h5py.File("/....../test_data.h5", 'r')
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
if os.path.exists('deepfuse.pt'):
    print('using pretrained model')
    state_dict = torch.load('deepfuse.pt',map_location='cuda')['model_state_dict']
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
        