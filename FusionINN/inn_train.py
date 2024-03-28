import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import h5py
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#############################################################################################################
#load train and validation data
hf = h5py.File("/projects/p084/p_discoret/Brats2018_training_data_sep_channels_train_val_mix.h5", 'r')
train_data = hf['data'][()]  # `data` is now an ndarray
hf.close()

hf = h5py.File("/projects/p084/p_discoret/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
val_data = hf['data'][()]  # `data` is now an ndarray
hf.close()

print("Training and Validation data has been loaded")

#############################################################################################################
#normalization
for i in range(len(train_data)):
    for j in range(4):
        train_data[i, j, :, :] = (train_data[i, j, :, :] - np.min(train_data[i, j, :, :])) / (np.max(train_data[i, j, :, :]) - np.min(train_data[i, j, :, :]))

for i in range(len(val_data)):
    for j in range(4):
        val_data[i, j, :, :] = (val_data[i, j, :, :] - np.min(val_data[i, j, :, :])) / (np.max(val_data[i, j, :, :]) - np.min(val_data[i, j, :, :]))

train_data_tensor = torch.from_numpy(train_data).float()
val_data_tensor = torch.from_numpy(val_data).float()

#############################################################################################################
#metaparameters
total_train_images = 8500
total_val_images = 1153
EPOCHS = 200
batch_size = 64
gpu_ids = [0,1]
lamda_ssim = 0.5
lamda_l2 = 0.5
lamda_l2_rev = 0.5
lamda_rev_ssim = 0.5
lamda_rev_fusion = 0.8
lamda_fusion = 0.8
lamda_total = 0.5

#############################################################################################################
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

##############################################################################################################
# Invertible Neural Network Architecture
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
#Definitions of the loss functions
#MMDLoss(Code:https://github.com/vislearn/analyzing_inverse_problems/blob/master/toy_8-modes/toy_8-modes.ipynb)
def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz
    XX, YY, XY = (torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda(),
                  torch.zeros(xx.shape).cuda())
    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1
    return torch.mean(XX + YY - 2.*XY)

     
l2_loss = torch.nn.MSELoss()

##############################################################################################################
#Optimizer, scheduler, DataParallel, checkpoint loads etc
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.95,patience=8,verbose=True)
state_dict=torch.load('/home/h1/s8993054/INN_Fusion/INN/Old_models/inn_400.pt',map_location='cuda')['model_state_dict']

model = model.float()
model = model.cuda() #all 8 GPUs will be used

model.load_state_dict(state_dict)

##############################################################################################################
#initialize lists for different loss functions
count = 0
loss_train = []
loss_ssim_train_t1ce = []
loss_ssim_train_flair = []
loss_ssim_train_combined = []

loss_l2_train_t1ce=[]
loss_l2_train_flair=[]
loss_l2_train_combined=[]

loss_latent_train = []

loss_ssim_rev_train_t1ce = []
loss_ssim_rev_train_flair = []
loss_ssim_rev_train_combined = []

loss_l2_rev_train_t1ce=[]
loss_l2_rev_train_flair=[]
loss_l2_rev_train_combined=[]

loss_val = []
loss_ssim_val_t1ce = []
loss_ssim_val_flair = []
loss_ssim_val_combined = []

loss_l2_val_t1ce=[]
loss_l2_val_flair=[]
loss_l2_val_combined=[]

loss_latent_val = []

loss_l2_rev_val_t1ce=[]
loss_l2_rev_val_flair=[]
loss_l2_rev_val_combined=[]

loss_ssim_rev_val_t1ce = []
loss_ssim_rev_val_flair = []
loss_ssim_rev_val_combined = []

ep_train_loss = []
ep_ssim_train_loss_t1ce = []
ep_ssim_train_loss_flair = []
ep_ssim_train_loss_combined = []
ep_l2_train_loss_t1ce=[]
ep_l2_train_loss_flair=[]
ep_l2_train_loss_combined=[]
ep_ssim_rev_train_loss_t1ce = []
ep_ssim_rev_train_loss_flair = []
ep_ssim_rev_train_loss_combined = []
ep_l2_rev_train_loss_t1ce=[]
ep_l2_rev_train_loss_flair=[]
ep_l2_rev_train_loss_combined=[]

ep_latent_train_loss = []

ep_val_loss = []
ep_ssim_val_loss_t1ce = []
ep_ssim_val_loss_flair = []
ep_ssim_val_loss_combined = []
ep_l2_val_loss_t1ce=[]
ep_l2_val_loss_flair=[]
ep_l2_val_loss_combined=[]
ep_l2_rev_val_loss_t1ce=[]
ep_l2_rev_val_loss_flair=[]
ep_l2_rev_val_loss_combined=[]
ep_ssim_rev_val_loss_t1ce = []
ep_ssim_rev_val_loss_flair = []
ep_ssim_rev_val_loss_combined = []

ep_latent_val_loss = []

#########################################################################################################################
#Start training
for epoch in range(EPOCHS):
    model.train()
    batch_idxs = total_train_images // batch_size
    for idx in range(batch_idxs):
        train_input = train_data_tensor[idx * batch_size:(idx + 1) * batch_size, 2:, :, :].cuda()
        
        #################################################################################################################
        #forward process
        #################################################################################################################
        (fused, z), log_jac_det = model(train_input)
        fused = torch.sigmoid(fused)
        #z = torch.sigmoid(z)
        z = z.reshape(batch_size, 240*240)
        count += 1
        ##################################### Fusion Loss ###############################################################
        # SSIM loss for the fusion training
        ssim_loss_t1ce = 1 - ssim(fused, train_input[:, 0:1, :, :], data_range=1)
        ssim_loss_flair = 1 - ssim(fused, train_input[:, 1:2, :, :], data_range=1)
        # club the T1ce and flair ssim losses
        ssim_loss_combined = lamda_ssim * ssim_loss_t1ce + (1 - lamda_ssim) * ssim_loss_flair
        # L2 loss for the fusion training
        l2_loss_t1ce = l2_loss(fused, train_input[:, 0:1, :, :])
        l2_loss_flair = l2_loss(fused, train_input[:, 1:2, :, :])
        # club the T1ce and flair L2 losses
        l2_loss_combined = lamda_l2 * l2_loss_t1ce + (1 - lamda_l2) * l2_loss_flair
        
        ##################################### Latent Loss ################################################################
        z_target = torch.randn(batch_size,240*240).cuda()
        latent_loss_train = MMD_multiscale(z, z_target)
        
        #########################Forward loss =  Fusion + Latent Loss ####################################################
        forward_loss_train=lamda_fusion * ssim_loss_combined + (1-lamda_fusion) * l2_loss_combined + latent_loss_train

        
        ##################################################################################################################
        #reverse process
        ##################################################################################################################
        z_train = torch.randn(batch_size,1,240,240).cuda()
        rev_input, _ = model([fused, z_train], rev=True)
        rev_input = torch.sigmoid(rev_input)
        
        ################################### Reconstruction Loss ##########################################################
        # SSIM loss for the reconstruction training
        ssim_loss_rev_t1ce = 1 - ssim(rev_input[:, 0:1, :, :], train_input[:, 0:1, :, :], data_range=1)
        ssim_loss_rev_flair = 1 - ssim(rev_input[:, 1:2, :, :], train_input[:, 1:2, :, :], data_range=1)
        # club the T1ce and flair SSIM losses
        ssim_loss_rev_combined = lamda_rev_ssim * ssim_loss_rev_t1ce + (1 - lamda_rev_ssim) * ssim_loss_rev_flair
        # L2 loss for the reconstruction training
        l2_rev_loss_t1ce = l2_loss(rev_input[:, 0:1, :, :], train_input[:, 0:1, :, :])
        l2_rev_loss_flair = l2_loss(rev_input[:, 1:2, :, :], train_input[:, 1:2, :, :])
        # club the T1ce and flair L2 losses
        l2_rev_loss_combined = lamda_l2_rev * l2_rev_loss_t1ce + (1 - lamda_l2_rev) * l2_rev_loss_flair
        # club the ssim and L2 Loss for the reconstruction training
        reconstruction_loss_train =lamda_rev_fusion * ssim_loss_rev_combined + (1-lamda_rev_fusion) * l2_rev_loss_combined
        
        ##################################################################################################################
        # combine the losses from the forward and reverse processes
        ##################################################################################################################
        train_loss_total = lamda_total * forward_loss_train + (1-lamda_total) * reconstruction_loss_train
        
        
        optimizer.zero_grad()
        train_loss_total.backward()
        optimizer.step()

        loss_train.append(train_loss_total.item())
        loss_ssim_train_t1ce.append(ssim_loss_t1ce.item())
        loss_ssim_train_flair.append(ssim_loss_flair.item())
        loss_l2_train_t1ce.append(l2_loss_t1ce.item())
        loss_l2_train_flair.append(l2_loss_flair.item())
        loss_ssim_rev_train_t1ce.append(ssim_loss_rev_t1ce.item())
        loss_ssim_rev_train_flair.append(ssim_loss_rev_flair.item())
        loss_l2_rev_train_t1ce.append(l2_rev_loss_t1ce.item())
        loss_l2_rev_train_flair.append(l2_rev_loss_flair.item())
        loss_l2_train_combined.append(l2_loss_combined.item())
        loss_l2_rev_train_combined.append(l2_rev_loss_combined.item())
        loss_ssim_train_combined.append(ssim_loss_combined.item())
        loss_ssim_rev_train_combined.append(ssim_loss_rev_combined.item())
        loss_latent_train.append(latent_loss_train.item())

    av_train_loss = np.average(loss_train)
    ep_train_loss.append(av_train_loss)

    av_ssim_train_t1ce = np.average(loss_ssim_train_t1ce)
    ep_ssim_train_loss_t1ce.append(av_ssim_train_t1ce)
    av_l2_train_t1ce=np.average(loss_l2_train_t1ce)
    ep_l2_train_loss_t1ce.append(av_l2_train_t1ce)
    av_l2_rev_train_t1ce=np.average(loss_l2_rev_train_t1ce)
    ep_l2_rev_train_loss_t1ce.append(av_l2_rev_train_t1ce)
    av_ssim_rev_train_loss_t1ce = np.average(loss_ssim_rev_train_t1ce)
    ep_ssim_rev_train_loss_t1ce.append(av_ssim_rev_train_loss_t1ce)

    av_ssim_train_loss_flair = np.average(loss_ssim_train_flair)
    ep_ssim_train_loss_flair.append(av_ssim_train_loss_flair)
    av_l2_train_loss_flair=np.average(loss_l2_train_flair)
    ep_l2_train_loss_flair.append(av_l2_train_loss_flair)
    av_l2_rev_train_loss_flair=np.average(loss_l2_rev_train_flair)
    ep_l2_rev_train_loss_flair.append(av_l2_rev_train_loss_flair)
    av_ssim_rev_train_loss_flair = np.average(loss_ssim_rev_train_flair)
    ep_ssim_rev_train_loss_flair.append(av_ssim_rev_train_loss_flair)

    av_ssim_train_loss_combined = np.average(loss_ssim_train_combined)
    ep_ssim_train_loss_combined.append(av_ssim_train_loss_combined)
    
    av_l2_train_loss_combined=np.average(loss_l2_train_combined)
    ep_l2_train_loss_combined.append(av_l2_train_loss_combined)
    
    av_l2_rev_train_loss_combined=np.average(loss_l2_rev_train_combined)
    ep_l2_rev_train_loss_combined.append(av_l2_rev_train_loss_combined)
    
    av_ssim_rev_train_loss_combined = np.average(loss_ssim_rev_train_combined)
    ep_ssim_rev_train_loss_combined.append(av_ssim_rev_train_loss_combined)
    
    av_latent_train_loss = np.average(loss_latent_train)
    ep_latent_train_loss.append(av_latent_train_loss)

    model.eval()
    val_batch_idxs = total_val_images // batch_size
    with torch.no_grad():
        for idx in range(val_batch_idxs):
            input_val = val_data_tensor[idx * batch_size:(idx + 1) * batch_size, 2:, :, :].cuda()
            
            #####################################################################################################################
            #forward process
            #####################################################################################################################
            (fused_val, z_val), log_jac_det_val = model(input_val)
            fused_val = torch.sigmoid(fused_val)          
            fused_val = fused_val.to(input_val.dtype)
            #z_val     = torch.sigmoid(z_val)
            z_val     = z_val.reshape(batch_size, 240*240)
            
            ###################################### Fusion Loss ##################################################################
            #ssim loss for fusion validation
            ssim_loss_t1ce_val = 1 - ssim(fused_val, input_val[:, 0:1, :, :], data_range=1)
            ssim_loss_flair_val = 1 - ssim(fused_val, input_val[:, 1:2, :, :], data_range=1)
            # club the T1ce and flair ssim losses
            ssim_loss_combined_val = lamda_ssim * ssim_loss_t1ce_val + (1 - lamda_ssim) * ssim_loss_flair_val
            # L2 loss for the  fusion validation
            l2_loss_t1ce_val= l2_loss(fused_val,input_val[:,0:1,...])
            l2_loss_flair_val= l2_loss(fused_val,input_val[:,1:2,...])
            # club the T1ce and flair L2 losses
            l2_loss_combined_val=lamda_l2*l2_loss_t1ce_val + (1-lamda_l2) * l2_loss_flair_val

            ######################################  Latent Loss #################################################################
            z_target_val = torch.randn(batch_size,240*240).cuda()
            latent_loss_val = MMD_multiscale(z_val, z_target_val)
            
            #########################Forward loss =  Fusion + Latent Loss #######################################################
            forward_loss_val=lamda_fusion * ssim_loss_combined_val + (1-lamda_fusion) * l2_loss_combined_val + latent_loss_val
            
            #####################################################################################################################
            #reverse process
            #####################################################################################################################
            z_val=torch.randn(batch_size,1,240,240).cuda()
            #z_val=torch.sigmoid(z_val)
            rev_val_input, _ = model([fused_val, z_val], rev=True)
            rev_val_input = torch.sigmoid(rev_val_input)
            
            # l2 loss for the fusion validation
            l2_rev_loss_t1ce_val = l2_loss(rev_val_input[:, 0:1, :, :], input_val[:, 0:1, :, :])
            l2_rev_loss_flair_val = l2_loss(rev_val_input[:, 1:2, :, :], input_val[:, 1:2, :, :])
            # club the T1ce and flair l2 losses
            l2_rev_loss_combined_val=lamda_l2_rev * l2_rev_loss_t1ce_val + (1-lamda_l2_rev) * l2_rev_loss_flair_val
            # SSIM loss for the reconstruction
            ssim_rev_loss_t1ce_val = 1 - ssim(rev_val_input[:, 0:1, :, :], input_val[:, 0:1, :, :], data_range=1)
            ssim_rev_loss_flair_val = 1 - ssim(rev_val_input[:, 1:2, :, :], input_val[:, 1:2, :, :], data_range=1)
            # club the T1ce and flair ssim losses
            ssim_rev_loss_combined_val = lamda_rev_ssim * ssim_rev_loss_t1ce_val + (1 - lamda_rev_ssim) * ssim_rev_loss_flair_val
            # club the ssim and L2 losses for the reconstruction
            reconstruction_loss_val=lamda_rev_fusion * ssim_rev_loss_combined_val + (1-lamda_rev_fusion) * l2_rev_loss_combined_val
            
            ######################################################################################################################
            # combine the losses from the forward and reverse processes 
            ######################################################################################################################
            validation_loss_total = lamda_total * forward_loss_val + (1-lamda_total) * reconstruction_loss_val  

            loss_val.append(validation_loss_total.item())
            loss_ssim_val_t1ce.append(ssim_loss_t1ce_val.item())
            loss_ssim_val_flair.append(ssim_loss_flair_val.item())
            loss_l2_val_t1ce.append(l2_loss_t1ce_val.item())
            loss_l2_val_flair.append(l2_loss_flair_val.item())
            loss_ssim_rev_val_t1ce.append(ssim_rev_loss_t1ce_val.item())
            loss_ssim_rev_val_flair.append(ssim_rev_loss_flair_val.item())
            loss_l2_rev_val_t1ce.append(l2_rev_loss_t1ce_val.item())
            loss_l2_rev_val_flair.append(l2_rev_loss_flair_val.item())
            loss_ssim_val_combined.append(ssim_loss_combined_val.item())
            loss_l2_val_combined.append(l2_loss_combined_val.item())
            loss_ssim_rev_val_combined.append(ssim_rev_loss_combined_val.item())
            loss_l2_rev_val_combined.append(l2_rev_loss_combined_val.item())
            loss_latent_val.append(latent_loss_val.item())
            
    scheduler.step(loss_val[-1])

    av_val_loss = np.average(loss_val)
    ep_val_loss.append(av_val_loss)

    av_ssim_val_loss_t1ce = np.average(loss_ssim_val_t1ce)
    ep_ssim_val_loss_t1ce.append(av_ssim_val_loss_t1ce)
    av_l2_val_loss_t1ce=np.average(loss_l2_val_t1ce)
    ep_l2_val_loss_t1ce.append(av_l2_val_loss_t1ce)
    av_l2_rev_val_loss_t1ce=np.average(loss_l2_rev_val_t1ce)
    ep_l2_rev_val_loss_t1ce.append(av_l2_rev_val_loss_t1ce)
    av_ssim_rev_val_loss_t1ce = np.average(loss_ssim_rev_val_t1ce)
    ep_ssim_rev_val_loss_t1ce.append(av_ssim_rev_val_loss_t1ce)

    av_ssim_val_loss_flair = np.average(loss_ssim_val_flair)
    ep_ssim_val_loss_flair.append(av_ssim_val_loss_flair)
    av_l2_val_loss_flair=np.average(loss_l2_val_flair)
    ep_l2_val_loss_flair.append(av_l2_val_loss_flair)
    av_l2_rev_val_loss_flair=np.average(loss_l2_rev_val_flair)
    ep_l2_rev_val_loss_flair.append(av_l2_rev_val_loss_flair)
    av_ssim_rev_val_loss_flair = np.average(loss_ssim_rev_val_flair)
    ep_ssim_rev_val_loss_flair.append(av_ssim_rev_val_loss_flair)

    av_ssim_val_loss_combined = np.average(loss_ssim_val_combined)
    ep_ssim_val_loss_combined.append(av_ssim_val_loss_combined)
    
    av_l2_val_loss_combined=np.average(loss_l2_val_combined)
    ep_l2_val_loss_combined.append(av_l2_val_loss_combined)
    
    av_l2_rev_val_loss_combined=np.average(loss_l2_rev_val_combined)
    ep_l2_rev_val_loss_combined.append(av_l2_rev_val_loss_combined)

    av_ssim_rev_val_loss_combined = np.average(loss_ssim_rev_val_combined)
    ep_ssim_rev_val_loss_combined.append(av_ssim_rev_val_loss_combined)
    
    av_latent_val_loss = np.average(loss_latent_val)
    ep_latent_val_loss.append(av_latent_val_loss)
#########################################################################################################################
#print the loss values
    print("Epoch: {}/200 LR:{} Train_loss:{} Val_loss:{} Train_SSIM_combined_loss:{} Val_SSIM_combined_loss:{} Train_Rev_SSIM_combined_loss:{}  Val_Rev_SSIM_combined_loss:{} Train_Latent_loss:{} Val_Latent_loss:{}"
    .format(epoch + 1,optimizer.param_groups[0]['lr'], ep_train_loss[-1], ep_val_loss[-1], ep_ssim_train_loss_combined[-1], ep_ssim_val_loss_combined[-1],
                ep_ssim_rev_train_loss_combined[-1],  ep_ssim_rev_val_loss_combined[-1], ep_latent_train_loss[-1], ep_latent_val_loss[-1]))
    torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss_total": ep_train_loss,
        "validation_loss_total": ep_val_loss,
        "training_loss_ssim_t1ce": ep_ssim_train_loss_t1ce,
        "validation_loss_ssim_t1ce": ep_ssim_val_loss_t1ce,
        "trainining_loss_l2_t1ce":ep_l2_train_loss_t1ce,
        "validation_loss_l2_t1ce":ep_l2_val_loss_t1ce,
        "training_loss_ssim_flair": ep_ssim_train_loss_flair,
        "validation_loss_ssim_flair": ep_ssim_val_loss_flair,
        "training_loss_l2_flair":ep_l2_train_loss_flair,
        "validation_loss_l2_flair":ep_l2_val_loss_flair,
        "training_loss_ssim_rev_t1ce": ep_ssim_rev_train_loss_t1ce,
        "validation_loss_ssim_rev_t1ce": ep_ssim_rev_val_loss_t1ce,
        "training_loss_l2_rev_t1ce":ep_l2_rev_train_loss_t1ce,
        "validation_loss_l2_rev_t1ce":ep_l2_rev_val_loss_t1ce,
        "training_loss_ssim_rev_flair": ep_ssim_rev_train_loss_flair,
        "validation_loss_ssim_rev_flair": ep_ssim_rev_val_loss_flair,
        "training_loss_l2_rev_flair":ep_l2_rev_train_loss_flair,
        "validation_loss_l2_rev_flair":ep_l2_rev_val_loss_flair,
        "training_loss_ssim_combined": ep_ssim_train_loss_combined,
        "validation_loss_ssim_combined": ep_ssim_val_loss_combined,
        "training_loss_l2_combined":ep_l2_train_loss_combined,
        "validation_loss_l2_combined":ep_l2_val_loss_combined,
        "training_loss_ssim_rev_combined": ep_ssim_rev_train_loss_combined,
        "validation_loss_ssim_rev_combined": ep_ssim_rev_val_loss_combined,
        "training_loss_l2_rev_combined":ep_l2_rev_train_loss_combined,
        "validation_loss_l2_rev_combined":ep_l2_rev_val_loss_combined,
        "training_loss_latent":ep_latent_train_loss,
        "validation_loss_latent": ep_latent_val_loss
    }, 'inn_600.pt')



