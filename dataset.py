# deep learning training package

def data_prep(task, batch_size):
    from torch.utils.data import DataLoader, Dataset
    import pandas as pd
    import os
    import numpy as np
    from torchvision import transforms, utils
    import torch 

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

            image = self.img_data[idx].astype(np.float32)

            if self.database == 'MIMIC':
                target = self.target_csv.iloc[idx, 3:]
            elif self.database == 'CheX':
                target = self.target_csv.iloc[idx, 5:]
            elif self.database == 'CXR8':
                b = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pneumonia', 'Pneumothorax']
                target = self.target_csv[b].iloc[idx]

            target = target.replace(np.nan, 0.0)
            target = target.replace(-1, 0.0)

            target = torch.tensor(target).type(torch.float)

            sample = {'image': image, 'target': target}

            if self.transform:
                sample['image'] = self.transform(sample['image'])

            return sample

    # load data and target
    print('Data Loading start!')
    print('---------------------------')

    if task == 'MIMIC':
        print('MIMIC task loading')
        raw_train_dataset = np.load('/data/che/MIMIC-CXR/train_test_val/train.npy')
        train_csv = pd.read_csv('/data/che/MIMIC-CXR/train_test_val/train.csv')

        raw_test_dataset = np.load('/data/che/MIMIC-CXR/train_test_val/test.npy')
        test_csv = pd.read_csv('/data/che/MIMIC-CXR/train_test_val/test.csv')
    
    elif task == 'CheX':
        print('CheX task loading')
        raw_train_dataset = np.load('/data/che/CheXpert/CheXpert-v1.0/CheXpert_train.npy')
        train_csv = pd.read_csv('/data/che/CheXpert/CheXpert-v1.0/CheXpert_train_no_normal.csv')

        raw_test_dataset = np.load('/data/che/CheXpert/CheXpert-v1.0/CheXpert_test.npy')
        test_csv = pd.read_csv('/data/che/CheXpert/CheXpert-v1.0/CheXpert_test_no_normal.csv')
        
    elif task == 'CXR8':
        print('CXR8 task loading')
        raw_train_dataset = np.load('/data/che/CXR8/DL_data_6_label/CXR8_train_6_label.npy')
        train_csv = pd.read_csv('/data/che/CXR8/DL_data_6_label/train_no_normal_6.csv')

        raw_test_dataset = np.load('/data/che/CXR8/DL_data_6_label/CXR8_test_6_label.npy')
        test_csv = pd.read_csv('/data/che/CXR8/DL_data_6_label/test_no_normal_6.csv')
    
    print(raw_train_dataset.shape, train_csv.shape)
    print(raw_test_dataset.shape, test_csv.shape)
    
    composed = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomAutocontrast(p=0.5)])   

    train_dataset = MIMIC_CheX_CXR8(img_data=raw_train_dataset, 
    target_csv=train_csv, database=task, transform=composed)

    test_dataset = MIMIC_CheX_CXR8(img_data=raw_test_dataset, 
    target_csv=test_csv, database=task, transform=composed)

    train_dataloader = DataLoader(dataset=train_dataset, 
    batch_size=batch_size, pin_memory=True, num_workers=0, shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset, 
    batch_size=256, pin_memory=True, shuffle=False)

    return train_dataloader, test_dataloader
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

# raw_train_dataset = np.load('/data/che/CheXpert/CheXpert-v1.0/CheXpert_train.npy')
# train_csv = pd.read_csv('/data/che/CheXpert/CheXpert-v1.0/CheXpert_train_no_normal.csv')


## CXR8 dataset loading

# raw_train_dataset = np.load('/data/che/CXR8/DL_data_6_label/CXR8_train_6_label.npy')
# train_csv = pd.read_csv('/data/che/CXR8/DL_data_6_label/train_no_normal_6.csv')

###-----------------------------------------------------------------------###

