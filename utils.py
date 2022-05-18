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
        return img.astype('uint8')





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
        return img.astype('uint8')

### CheXpert Database
def get_CheXpert_img(img_path, grey_scale=False, normalize=255, resize=256):
    # database info csv path:
    # '/data/che/CheXpert/CheXpert-v1.0'
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
        return img.astype('uint8')


### MIMIC-CXR Database
def get_MIMIC_img(subject_id, study_id, grey_scale=False, normalize=255, resize=256):
    # database info csv path:
    # '/data/che/MIMIC-CXR/files/'
    # subject_id: ~, study_id: ~
    # original intensity: 255
    # original resolution 3000+

    path ='/data/che/MIMIC-CXR/files/'
    sub_dir = 'p' + subject_id[0:2] + '/' + 'p' + subject_id + '/' + 's' + study_id + '/'
    jpg_path = path + sub_dir
    jpg_list = os.listdir(jpg_path)

    # generate a empty list for jpg saving
    img_list = [None for _ in range(len(jpg_list))]

    for i, jpg_name in enumerate(jpg_list):
        img_path = jpg_path + jpg_name

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
            continue
        else:
            img = img.astype('uint8')

        # save jpg array into list
        img_list[i] = img
    
    # return a list of jpgs
    return img_list