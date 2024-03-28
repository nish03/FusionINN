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


##############################################################################################################
# Invertible Neural Network Architecture
##############################################################################################################
#subnet architectures
def subnet_conv1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32,  c_out, 3, padding=1), nn.ReLU())
def subnet_conv2(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(64,  c_out, 3, padding=1), nn.ReLU())
def subnet_conv3(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32,  c_out, 3, padding=1), nn.ReLU())


nodes = [Ff.InputNode(2,240,240, name='Input Img')]
nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}, name='ActNorm'))
nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                         {'subnet_constructor':subnet_conv1, 'clamp':2.0},
                         name='conv_1'))
nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {}, name='Permutation'))
nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='DownSampling'))
nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                         {'subnet_constructor':subnet_conv2, 'clamp':2.0},
                         name='conv_2'))
nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {}, name='Permutation'))
nodes.append(Ff.Node(nodes[-1], Fm.IRevNetUpsampling, {}, name='UpSampling'))
nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                         {'subnet_constructor':subnet_conv3, 'clamp':2.0},
                         name='conv_3'))
split_node = Ff.Node(nodes[-1], Fm.Split, {}, name='Split')
nodes.append(split_node)
output1 = Ff.OutputNode(split_node.out0, name='Output 1')
output2 = Ff.OutputNode(split_node.out1, name='Output 2')
nodes.append(output1)
nodes.append(output2)

model = Ff.GraphINN(nodes)
print(model)

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
#torch.save(val_data_tensor, 'val_data_tensor.pt')

# load pretrain model
if os.path.exists('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt'):
    print('using pretrained model')
    state_dict = torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn.pt',map_location='cuda')['model_state_dict']
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
recon_val = torch.zeros(total_val_images, 2, 240, 240)
count = 0
with torch.no_grad():
    for idx in range(total_val_images):
        test_val = val_data_tensor[idx:idx+1,2:,:,:].to(device)
        (fused_test, z_test), log_jac_det_val = model(test_val)
        fused_test = torch.sigmoid(fused_test)
        fused_val[idx,] = fused_test[0,]

        z_test = torch.randn(1, 1, 240, 240).cuda()
        rev_test_input, _ = model([fused_test, z_test], rev=True)
        rev_test_input = torch.sigmoid(rev_test_input)
        recon_val[idx,] = rev_test_input[0,]
        count += 1
        if count % 100 == 0:
            print(count)
    torch.save(fused_val, 'val_fused_tensor.pt')
    torch.save(recon_val, 'val_recon_tensor.pt')
        