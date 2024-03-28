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
hf = h5py.File("/projects/p084/p_discoret/Brats2018_training_data_sep_channels_train_val_mix.h5", 'r')
train_data = hf['data'][()]  # `data` is now an ndarray
hf.close()

hf = h5py.File("/projects/p084/p_discoret/Brats2018_validation_data_sep_channels_train_val_mix.h5", 'r')
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

################################################################################################################
#Definitions of the loss functions   
l2_loss = torch.nn.MSELoss()

##############################################################################################################
#Optimizer, scheduler, DataParallel, checkpoint loads etc
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.95,patience=8,verbose=True)
#state_dict=torch.load('rev_inn_model_l2.pt',map_location='cuda')['model_state_dict']

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
    }, 'unet.pt')
