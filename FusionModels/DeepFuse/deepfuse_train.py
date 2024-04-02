import torch
import torch.nn as nn
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import h5py
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##################################################################################################################
#load train and validation data
hf = h5py.File("/....../train_data.h5", 'r')
train_data = hf['data'][()]  # `data` is now an ndarray
hf.close()

hf = h5py.File("/....../test_data.h5", 'r')
val_data = hf['data'][()]  # `data` is now an ndarray
hf.close()

print("Training and Validation data has been loaded")

##################################################################################################################
#normalization
for i in range(len(train_data)):
    for j in range(4):
        train_data[i, j, :, :] = (train_data[i, j, :, :] - np.min(train_data[i, j, :, :])) / (np.max(train_data[i, j, :, :]) - np.min(train_data[i, j, :, :]))

for i in range(len(val_data)):
    for j in range(4):
        val_data[i, j, :, :] = (val_data[i, j, :, :] - np.min(val_data[i, j, :, :])) / (np.max(val_data[i, j, :, :]) - np.min(val_data[i, j, :, :]))

train_data_tensor = torch.from_numpy(train_data).float()
val_data_tensor = torch.from_numpy(val_data).float()

##################################################################################################################
#metaparameters
total_train_images = 8500
total_val_images = 1153
EPOCHS = 400
batch_size = 64
gpu_ids = [0,1]
lamda_ssim = 0.5
lamda_l2 = 0.5
lamda_fusion = 0.8

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

#################################################################################################################
#Definitions of the loss functions   
l2_loss = torch.nn.MSELoss()

##############################################################################################################
#Optimizer, scheduler, DataParallel, checkpoint loads etc
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.95,patience=8,verbose=True)

##############################################################################################################
#initialize lists for different loss functions
count = 0

loss_train = []
loss_ssim_train_t1ce = []
loss_ssim_train_flair = []
loss_l2_train_t1ce=[]
loss_l2_train_flair=[]
loss_ssim_train_combined = []
loss_l2_train_combined=[]

loss_val = []
loss_ssim_val_t1ce = []
loss_ssim_val_flair = []
loss_l2_val_t1ce=[]
loss_l2_val_flair=[]
loss_ssim_val_combined = []
loss_l2_val_combined=[]

ep_train_loss = []
ep_ssim_train_loss_t1ce = []
ep_ssim_train_loss_flair = []
ep_l2_train_loss_t1ce=[]
ep_l2_train_loss_flair=[]
ep_ssim_train_loss_combined = []
ep_l2_train_loss_combined=[]

ep_val_loss = []
ep_ssim_val_loss_t1ce = []
ep_ssim_val_loss_flair = []
ep_l2_val_loss_t1ce=[]
ep_l2_val_loss_flair=[]
ep_ssim_val_loss_combined = []
ep_l2_val_loss_combined=[]

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
        fused = model(train_input[:,0:1,:,:], train_input[:,1:,:,:])
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
        
        train_loss_total=lamda_fusion * ssim_loss_combined + (1-lamda_fusion) * l2_loss_combined 

        optimizer.zero_grad()
        train_loss_total.backward()
        optimizer.step()

        loss_train.append(train_loss_total.item())
        loss_ssim_train_t1ce.append(ssim_loss_t1ce.item())
        loss_ssim_train_flair.append(ssim_loss_flair.item())
        loss_ssim_train_combined.append(ssim_loss_combined.item())
        loss_l2_train_t1ce.append(l2_loss_t1ce.item())
        loss_l2_train_flair.append(l2_loss_flair.item())
        loss_l2_train_combined.append(l2_loss_combined.item())

    av_train_loss = np.average(loss_train)
    ep_train_loss.append(av_train_loss)

    av_ssim_train_t1ce = np.average(loss_ssim_train_t1ce)
    ep_ssim_train_loss_t1ce.append(av_ssim_train_t1ce)
    
    av_l2_train_t1ce=np.average(loss_l2_train_t1ce)
    ep_l2_train_loss_t1ce.append(av_l2_train_t1ce)

    av_ssim_train_loss_flair = np.average(loss_ssim_train_flair)
    ep_ssim_train_loss_flair.append(av_ssim_train_loss_flair)
    
    av_l2_train_loss_flair=np.average(loss_l2_train_flair)
    ep_l2_train_loss_flair.append(av_l2_train_loss_flair)

    av_ssim_train_loss_combined = np.average(loss_ssim_train_combined)
    ep_ssim_train_loss_combined.append(av_ssim_train_loss_combined)
    
    av_l2_train_loss_combined=np.average(loss_l2_train_combined)
    ep_l2_train_loss_combined.append(av_l2_train_loss_combined)

    model.eval()
    val_batch_idxs = total_val_images // batch_size
    with torch.no_grad():
        for idx in range(val_batch_idxs):
            input_val = val_data_tensor[idx * batch_size:(idx + 1) * batch_size, 2:, :, :].cuda()
            
            #####################################################################################################################
            #forward process
            #####################################################################################################################
            fused_val = model(input_val[:,0:1,:,:], input_val[:,1:,:,:])
            
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

            validation_loss_total=lamda_fusion * ssim_loss_combined_val + (1-lamda_fusion) * l2_loss_combined_val 
            
            loss_val.append(validation_loss_total.item())
            loss_ssim_val_t1ce.append(ssim_loss_t1ce_val.item())
            loss_ssim_val_flair.append(ssim_loss_flair_val.item())
            loss_l2_val_t1ce.append(l2_loss_t1ce_val.item())
            loss_l2_val_flair.append(l2_loss_flair_val.item())
            loss_ssim_val_combined.append(ssim_loss_combined_val.item())
            loss_l2_val_combined.append(l2_loss_combined_val.item())
            
    scheduler.step(loss_val[-1])

    av_val_loss = np.average(loss_val)
    ep_val_loss.append(av_val_loss)

    av_ssim_val_loss_t1ce = np.average(loss_ssim_val_t1ce)
    ep_ssim_val_loss_t1ce.append(av_ssim_val_loss_t1ce)
    av_l2_val_loss_t1ce=np.average(loss_l2_val_t1ce)
    ep_l2_val_loss_t1ce.append(av_l2_val_loss_t1ce)

    av_ssim_val_loss_flair = np.average(loss_ssim_val_flair)
    ep_ssim_val_loss_flair.append(av_ssim_val_loss_flair)
    av_l2_val_loss_flair=np.average(loss_l2_val_flair)
    ep_l2_val_loss_flair.append(av_l2_val_loss_flair)

    av_ssim_val_loss_combined = np.average(loss_ssim_val_combined)
    ep_ssim_val_loss_combined.append(av_ssim_val_loss_combined)
    
    av_l2_val_loss_combined=np.average(loss_l2_val_combined)
    ep_l2_val_loss_combined.append(av_l2_val_loss_combined)
    
#########################################################################################################################
#print the loss values
    print("Epoch: {}/400 LR:{} Train_loss:{} Val_loss:{} Train_SSIM_combined_loss:{} Val_SSIM_combined_loss:{}"
    .format(epoch + 1,optimizer.param_groups[0]['lr'], ep_train_loss[-1], ep_val_loss[-1], ep_ssim_train_loss_combined[-1], ep_ssim_val_loss_combined[-1]))
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
        "training_loss_ssim_combined": ep_ssim_train_loss_combined,
        "validation_loss_ssim_combined": ep_ssim_val_loss_combined,
        "training_loss_l2_combined":ep_l2_train_loss_combined,
        "validation_loss_l2_combined":ep_l2_val_loss_combined,
    }, 'deepfuse.pt')
