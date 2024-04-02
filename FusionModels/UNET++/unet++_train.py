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

#############################################################################################################
#load train and validation data
hf = h5py.File("/......../train_data.h5", 'r')
train_data = hf['data'][()]  # `data` is now an ndarray
hf.close()

hf = h5py.File("/......../test_data.h5", 'r')
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
EPOCHS = 400
batch_size = 64
gpu_ids = [0,1]
lamda_ssim = 0.5
lamda_l2 = 0.5
lamda_fusion = 0.8

#############################################################################################################
#UNet++ Architecture
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
        
        
class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [ 64, 128, 256, 512,1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return nn.Sigmoid()(output)
            
    
model= NestedUNet().to(device)
model = model.float()
model=torch.nn.DataParallel(model, gpu_ids)

###############################################################################################################
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
        fused = model(train_input)
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
            fused_val = model(input_val)
            
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
    }, 'unet++.pt')




