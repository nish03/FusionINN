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
#UNET Architecture
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

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(2, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs
        
model= build_unet().to(device)
model = model.float()
model=torch.nn.DataParallel(model, gpu_ids) 

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
if os.path.exists('unet.pt'):
    print('using pretrained model')
    state_dict = torch.load('unet.pt',map_location='cuda')['model_state_dict']
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
        