import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import h5py
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.utils.tensorboard import SummaryWriter


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#clinical_fused_tensor = torch.load("/home/h1/s8993054/INN_Fusion/Clinical_Fusion/clinical_fused_tensor_unet3+.pt")
#clinical_fused_numpy  = clinical_fused_tensor.cpu().squeeze()

clinical_fused_tensor = torch.zeros(6,1,240,240)
for i in range(6):
        fused_img = cv2.imread('/home/h1/s8993054/INN_Fusion/DDFM/output/clinical_output/' + str(i) + '.png').astype('float32')
        fused_img = np.round(cv2.cvtColor(fused_img, cv2.COLOR_BGR2GRAY))
        fused_img = fused_img[np.newaxis,np.newaxis, ...]
        fused_img = fused_img/255.0 
        fused_img = fused_img.astype('float32')
        clinical_fused_tensor[i,] = torch.FloatTensor(fused_img[0,])

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

###############################################################################################################
# Load the data and the model
###############################################################################################################
clinical_fused_tensor = clinical_fused_tensor.float()

# load pretrain model
print('using pretrained model')
state_dict = torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn_400.pt',map_location='cuda')['model_state_dict']
model.load_state_dict(state_dict)
model.to(device)
model = model.float()

###############################################################################################################
# start the testing loop
###############################################################################################################
model.eval()
recon_val = torch.zeros(clinical_fused_tensor.shape[0], 2, 240, 240)
count = 0
with torch.no_grad():
    for idx in range(clinical_fused_tensor.shape[0]):
        z_test = torch.randn(1, 1, 240, 240).cuda()
        rev_test_input, _ = model([clinical_fused_tensor[idx:idx+1,].cuda(), z_test], rev=True)
        rev_test_input = torch.sigmoid(rev_test_input)
        recon_val[idx,] = rev_test_input[0,]
        count += 1
        if count % 100 == 0:
            print(count)   
    torch.save(recon_val, 'clinical_recon_tensor_DDFM_on_INN.pt')
    
clinical_recon_numpy = recon_val.cpu().squeeze()

    
#save the clinical images
for i in range(clinical_fused_tensor.shape[0]):
    plt.imsave('/home/h1/s8993054/INN_Fusion/Clinical_Fusion/Recon_DDFM_Flair/Recon_Flair_' + str(i) + '.png', clinical_recon_numpy[i,0,], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.clf()
    plt.imsave('/home/h1/s8993054/INN_Fusion/Clinical_Fusion/Recon_DDFM_DWI/Recon_DWI_' + str(i) + '.png', clinical_recon_numpy[i,1,], cmap = 'gray', vmin = 0.0, vmax = 1.0)
    plt.close()
        