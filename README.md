# FusionINN
This repository provides the source code for the paper titled [FusionINN: Invertible Image Fusion for Brain Tumor Monitoring](https://arxiv.org/abs/2403.15769). 

![GitHub Logo](/assets/Method.png)

**The key contributions of the paper are as follows:**
* Introduces first-of-its-kind image fusion framework that harnesses invertible normalizing flow for bidirectional training. 
* The framework not only generates a fused image but can also decompose it into constituent source images, thus enhancing the interpretability for clinical practitioners. 
* Conducts evaluation studies that shows state-of-the-art results with standard fusion metrics, alongside its additional capability to decompose the fused images.
* Illustrates the framework's clinical viability by effectively decomposing and fusing new images from source modalities not encountered during training. 

# Advertisement
* Check out our AAAI 2024 work [QuantOD](https://github.com/taghikhah/QuantOD) on Outlier-aware Image Classification.
* Check out our CVPR 2023 highlight work [FFS](https://github.com/nish03/FFS/) on Outlier-aware Object Detection.
  
# Content
* [How to Cite](#how-to-cite)
* [Installation](#installation)
  * [Package Requirements](#package-requirements)
  * [Datasets](#datasets)
* [Training FFS from scratch](#training-ffs-from-scratch)
* [Using Pre-trained FFS Models](#using-pre-trained-ffs-models)
* [Inference procedure](#inference-procedure)
* [Visualization of results](#visualization-of-results)
* [License](#license)

# How to Cite
If you find this code or paper useful in your research, please consider citing our paper as follows:
```
@misc{kumar2024fusioninn,
      title={FusionINN: Invertible Image Fusion for Brain Tumor Monitoring}, 
      author={Nishant Kumar and Ziyan Tao and Jaikirat Singh and Yang Li and Peiwen Sun and Binghui Zhao and Stefan Gumhold},
      year={2024},
      eprint={2403.15769},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

# Installation

## Package Requirements
```
pip install -r requirements.txt
```

## Datasets
We provide a processed version of the [BraTS 2018 data](https://www.med.upenn.edu/sbia/brats2018/data.html) used to train the FusionINN framework and other evaluated fusion models. The processed data only contain those images from the BraTS 2018 dataset where the clinical annotations shows the presence of the necrotic core, non-enhancing tumor and peritumoral edema. 
The processed data consists of roughly 10000 image pairs, which we shuffled and partitioned it into training and test sets. The processed data can be downloaded from [link for test set](https://datashare.tu-dresden.de/s/8ZRbWJNMQnftDRy) and [link for training set](. 

**Note:** Please be aware that if you use this data for your own research, you need to cite the original manuscripts as stated at the [BraTS 2018 page](https://www.med.upenn.edu/sbia/brats2018/data.html).

# Training FusionINN framework
**Step 1:** First and foremost, downloaed the FusionINN project repository and make sure you are inside the project folder by running
```
cd /path/to/FusionINN-main/FusionINN 
```

**Step 2:** Make sure you change the folder path where the dataset is placed in ```inn_train.py```, before you start the training process. Also, ensure that you have allocated atleast 1 GPU for the training. Then, run the following command:

```
python inn_train.py  
```
The trained model will be saved at ```/path/to/FusionINN-main/FusionINN/inn.pt```.

# Training other Fusion Models
**Step 1:** Make sure you are inside the correct project folder. For example: the path for the DeepFuse model will be as follows:
```
cd /path/to/FusionINN-main/FusionModels/DeepFuse 
```

**Step 2:** Make sure you change the folder path where the dataset is placed in ```deepfuse_train.py```, before you start the training process. Also, ensure that you have allocated atleast 1 GPU for the training. Then, run the following command:

```
python deepfuse_train.py  
```
The trained model will be saved at ```/path/to/FusionINN-main/FusionModels/DeepFuse/deepfuse.pt```. Please follow the same proedure for other fusion models.


# Using Pre-trained FusionINN Model
If you would like to directly use the pre-trained FusionINN model instead of training a new instance of the FusionINN, please follow the below steps:

**Step 1:** You need to download the pre-trained FusionINN model from [here](https://datashare.tu-dresden.de/s/xQPDgiLRQkeT6eJ). 

**Step 2:** Place ```inn.pt``` in the exact same folder where the trained model gets saved for the training procedure i.e. ```/path/to/FusionINN-main/FusionINN/```. 


# Using Pre-trained DDFM Model
Please note that you need to directly use a pre-trained Diffusion model to run DDFM approach as this method is not adaptable for training from scratch. Hence, to test DDFM approach, please follow the below steps:

**Step 1:** You need to download the pre-trained model named ```256x256_diffusion_uncond.pt``` from [here](https://github.com/openai/guided-diffusion) and place it in ```/path/to/FusionINN-main/FusionModels/DDFM/output/``` folder. 

**Step 2:** Run the follwoing command:

```
cd /path/to/FusionINN-main/FusionModels/DDFM/
python sample_brats.py
```

The above steps will save the fused images obtained from DDFM model in the following path ```/path/to/FusionINN-main/FusionModels/DDFM/output/recon/```. 


# Inference procedure
Make sure the paths are correct in the file ```inn_test.py```. The inference procedure is common irrespective of whether you use pre-trained models or trained the models following the procedure mentioned before. For the FusionINN model as an example, run the follwoing command:

```
cd /path/to/FusionINN-main/FusionINN 
python inn_test.py
```

The files namely ```val_fused_tensor.pt``` and  ```val_recon_tensor.pt``` will be saved at ```/path/to/FusionINN-main/FusionINN/```.

**Note:** Please modify the inference procedure according to the model you want to test. Please note that other models will only produce ```val_fused_tensor.pt``` file since they are not invertible. 


# Visualization of FusionINN results
To visualize the fusion and decomposition performance of FusionINN, please follow the below steps: 

**Step 1:** Create five new folders two for input images (named ```T1ce``` and ```Flair```), one for fused images (named ```Fused```) and two for decomposed images (named ```Recon_T1ce``` and ```Recon_Flair```) in the folder ```/path/to/FusionINN-main/FusionINN```. 

**Step 2:** Run the following command: 

```
cd /path/to/FusionINN-main/FusionINN 
python inn_vis.py
```

**Note:** Please modify the visualize procedure according to the model you want to test. Please note that other models will not require folders for decomposed images. 

# Computing Quantitative Results

**Requirement:** Please make sure you have MATLAB installed in your workspace. Please follow the below steps:

**Step 1:** To compute the SSIM metric scores, run the following command:

```
cd /path/to/FusionINN-main/ 
python ssim_test.py
```

**Step 2:** Enter the MATLAB environment and add all Folders and the Subfolders of FusionINN project folder to search path of your MATLAB ennvironment. Next, running the ```evaluate.m``` file in te path ```/path/to/FusionINN-main/``` will save a file ```Q.mat``` in the same folder path. The ```Q.mat``` file will contain scores obtained from four other fusion metrics for an evaluated fusion model.

**Step 3:** Finally run the following command to obtain mean values of all the five fusion metrics for all the evaluated models:

```
cd /path/to/FusionINN-main/ 
python mean_test.py
```


# License
This software is licensed under the MIT license. 






