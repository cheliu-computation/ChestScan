import os

# preprocess & postprocess package
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from skimage import io, transform
import skimage
from PIL import Image

# deep learning training package
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

### PadChest Database
def get_PadChest_img(img_name, grey_scale=False, normalize=255, resize=256):
    # database info csv path:
    # '/data/che/PadChest/BIMCV-PadChest-FULL/labels/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'
    # img_name: png file name
    # original intensity: 6e4+
    # original resolution 3000+

    path = '/data/che/PadChest/BIMCV-PadChest-FULL/PadChestImage/'
    img_path = path + img_name
    img = Image.open(img_path)

    # convert png file to numpy array
    img = np.array(img)
    # rescale intensity to 0~normalize
    img = ((img - img.min()) * (1/(img.max() - img.min()) * normalize))

    if grey_scale:
        # convert 1 channel to 3 channel(not change image, just visulize like gray scale)
        img = np.expand_dims(img, -1)
        img = np.repeat(img, 3, axis=-1)

        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    else:
        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    
    if normalize == 1:
        return img
    else:
    	if grey_scale:
    		return img.astype('uint8')
        else:
        	return img


###-----------------------------------------------------------------------###

### CXR8 Database
def get_CXR8_img(img_name, grey_scale=False, normalize=255, resize=256):

    # database info csv path:
    # '/data/che/CXR8/Data_Entry_2017_v2020.csv
    # img_name: png file name
    # original intensity: 255
    # original resolution 1024

    path = '/data/che/CXR8/images/images/'
    img_path = path + img_name
    img = Image.open(img_path)
    
    # convert png file to numpy array
    img = np.array(img)

    # rescale intensity to 0~normalize
    img = ((img - img.min()) * (1/(img.max() - img.min()) * normalize))

    if grey_scale:
        # convert 1 channel to 3 channel(not change image, just visulize like gray scale)
        img = np.expand_dims(img, -1)
        img = np.repeat(img, 3, axis=-1)
        
        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    else:
        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)

    if normalize == 1:
        return img
    else:
    	if grey_scale:
    		return img.astype('uint8')
        else:
        	return img

### CXR8 Dataset
class CXR8(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_data, target_csv, transform=None):
        """
        Args:
            img_data (numpy array-(idx, 256, 256)): resized, normalized image data.
            target_csv (csv file): dicom_id, subject_id, study_id, 12 labels exclude No Finding.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.img_data = img_data
        self.target_csv = target_csv
        self.transform = transform
        # self.target_csv = pd.read_csv(target_csv)

    def __len__(self):
        return (self.target_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.img_data[idx]
        
        target = self.target_csv['One-hot Labels'][idx]
        target = np.fromstring(target[1:-1], dtype=int, sep=',')
        target = torch.tensor(target).type(torch.float)

        sample = {'image': image, 'target': target}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

### code demo for CXR8 dataset and dataloader

# raw_train_dataset = np.load('/data/che/CXR8/DL_data/CXR8_train.npy')
# train_csv = pd.read_csv('/data/che/CXR8/DL_data/train_no_normal.csv')
# raw_test_dataset = np.load('/data/che/CXR8/DL_data/CXR8_test.npy')
# test_csv = pd.read_csv('/data/che/CXR8/DL_data/test_no_normal.csv')

# train_dataset = CXR8(img_data=raw_train_dataset, target_csv=train_csv, transform=composed)
# test_dataset = CXR8(img_data=raw_test_dataset, target_csv=test_csv, transform=composed)

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, pin_memory=True, num_workers=8, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, pin_memory=False, num_workers=8, shuffle=False)

### split and extract label from CXR8 csv
def spl_ext(label_series):
	return label_series.str.slice(1, -1).str.replace('\'','', regex=True).str.split(', ')
###-----------------------------------------------------------------------###

### CheXpert Database
def get_CheXpert_img(img_path, grey_scale=False, normalize=255, resize=256):
    # database info csv path:
    # '/data/che/CheXpert/CheXpert-v1.0/train.csv'
    # '/data/che/CheXpert/CheXpert-v1.0/test.csv'
    # img_name: png file name
    # original intensity: 255
    # original resolution 2000+

    path = '/data/che/CheXpert/'
    img_path = path + img_path
    img = Image.open(img_path)

    # convert png file to numpy array
    img = np.array(img)

    # rescale intensity to 0~normalize
    img = ((img - img.min()) * (1/(img.max() - img.min()) * normalize))

    if grey_scale:
        # convert 1 channel to 3 channel(not change image, just visulize like gray scale)
        img = np.expand_dims(img, -1)
        img = np.repeat(img, 3, axis=-1)

        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    else:
        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    
    
    if normalize == 1:
        return img
    else:
        if grey_scale:
            return img.astype('uint8')
        else:
            return img


### MIMIC-CXR Database
def get_MIMIC_img(subject_id, study_id, dicom, grey_scale=False, normalize=255, resize=256):
    # database info csv path:
    # '/data/che/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv'
    # img_name: png file name

    path ='/data/che/MIMIC-CXR/files/'
    sub_dir = 'p' + subject_id[0:2] + '/' + 'p' + subject_id + '/' + 's' + study_id + '/' + dicom + '.jpg'
    jpg_path = path + sub_dir

    img = Image.open(jpg_path)

    # convert png file to numpy array
    img = np.array(img)

    # rescale intensity to 0~normalize
    img = ((img - img.min()) * (1/(img.max() - img.min()) * normalize))

    if grey_scale:
        # convert 1 channel to 3 channel(not change image, just visulize like gray scale)
        img = np.expand_dims(img, -1)
        img = np.repeat(img, 3, axis=-1)

        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    else:
        # resize image 
        img = skimage.transform.resize(img, (resize, resize), 
        order=1, preserve_range=True, anti_aliasing=False)
    

    if normalize == 1:
        return img.astype('float16')
    else:
        if grey_scale:
            return img.astype('uint8')
        else:
            return img.astype('float16')

### MIMIC_CheX Dataset
### MIMIC_CheX Dataset
class MIMIC_CheX_CXR8(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_data, target_csv, database, transform=None):
        """
        Args:
            img_data (numpy array-(idx, 256, 256)): resized, normalized image data.
            target_csv (csv file): dicom_id, subject_id, study_id, 12 labels exclude No Finding.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.img_data = img_data
        self.target_csv = target_csv
        self.transform = transform
        self.database = database

    def __len__(self):
        return (self.target_csv.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.img_data[idx]
        
        if self.database == 'MIMIC':
            target = self.target_csv.iloc[idx, 3:]
        elif self.database == 'CheX':
            target = self.target_csv.iloc[idx, 5:]
        elif self.database == 'CXR8':
            target = self.target_csv.iloc[idx, 3:]
            
        target = target.replace(np.nan, 0.0)
        target = torch.tensor(target).type(torch.float)

        sample = {'image': image, 'target': target}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

### code demo for MIMIC-CXR/CheXpert/CXR8 dataset and dataloader

## MIMIC-CXR dataset loading

# raw_train_dataset = np.load('/data/che/MIMIC-CXR/train_test_val/train.npy')
# train_csv = pd.read_csv('/data/che/MIMIC-CXR/train_test_val/train.csv')

# raw_test_dataset = np.load('/data/che/MIMIC-CXR/train_test_val/test.npy')
# test_csv = pd.read_csv('/data/che/MIMIC-CXR/train_test_val/test.csv')

# raw_val_dataset = np.load('/data/che/MIMIC-CXR/train_test_val/val.npy')
# val_csv = pd.read_csv('/data/che/MIMIC-CXR/train_test_val/val.csv')

# train_dataset = MIMIC_CheX(img_data=raw_train_dataset, target_csv=train_csv, transform=composed)
# test_dataset = MIMIC_CheX(img_data=raw_test_dataset, target_csv=test_csv, transform=composed)
# val_dataset = MIMIC_CheX(img_data=raw_val_dataset, target_csv=val_csv, transform=composed)

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, pin_memory=True, num_workers=8, shuffle=True)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, pin_memory=False, num_workers=8, shuffle=False)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, pin_memory=False, num_workers=8, shuffle=False)

## CheXpert dataset loading

# raw_train_dataset = np.load('/data/che/MIMIC-CXR/train_test_val/train.npy')
# train_csv = pd.read_csv('/data/che/MIMIC-CXR/train_test_val/train.csv')

## CXR8 dataset loading

# raw_train_dataset = np.load('/data/che/CXR8/DL_data_6_label/CXR8_train_6_label.npy')
# train_csv = pd.read_csv('/data/che/CXR8/DL_data_6_label/train_no_normal_6.csv')

###-----------------------------------------------------------------------###


'''
intersected label: ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pneumonia', 'Pneumothorax']
difference label: ['Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening']
'''