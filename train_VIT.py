# Copyright (c) 2022 Unnamed Network for ECG classification
# Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training process code for Unnamed Netowrk on HPC.
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns
# import torch package
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, plot_confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset
# import assistant package
from tqdm import tqdm
import wandb

import torch_optimizer as optim
from vit_pytorch import ViT


from dataset_VIT import data_prep

# check device available
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device_id = [i for i in range(torch.cuda.device_count())]
        device = 'cuda'
    else:
        print("Let's use 1 GPU!")
        device = 'cuda'
else:
    device = 'cpu'

# release the GPU
torch.cuda.empty_cache()
wandb.init(settings=wandb.Settings(start_method='fork'))

def setup_seed(seed):
    # paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def start_train(args, model, train_loader, test_loader, device):
    # learning rate decay and optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
    # weight_decay= 2e-5)
    optimizer = optim.Lamb(model.parameters(), lr=args.lr, weight_decay=2e-5)

    warmup_it = 400
    lr_min, lr_max = args.lr/(2**5), args.lr
    T_max = 400
    lambda1 = lambda epoch: lr_max * epoch / warmup_it if epoch < warmup_it else (
        lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(np.pi * (epoch-warmup_it)/T_max))
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda1], verbose=True)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
    #  factor=0.5, patience=4, min_lr=args.lr/16, threshold=2)
    
    # loss function
    criterion = nn.CrossEntropyLoss()

    print('Start training model...')
    print('---------------------------')
    def train(model, criterion, optimizer, train_loader, device, epoch):
        epoch_loss = 0.0
        model.train()
        # n= 0
        for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['target']
            inputs = inputs.to(device).contiguous()
            labels = labels.to(device).contiguous()
            optimizer.zero_grad()

            with autocast():
                
                outputs = model(inputs)
                # outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels)

            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            # n+=1
            # if n>3:
            #     break
        return model, optimizer, epoch_loss

    @torch.no_grad()
    def test(model, criterion, test_loader, device):
        model.eval()
        test_loss = 0 
        
        output_prob = [None for _ in range(len(test_loader))]
        targets = [None for _ in range(len(test_loader))]

        for idx, data in enumerate(test_loader):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['target']
            inputs = inputs.to(device).contiguous()
            labels = labels.to(device).contiguous()

            outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)

            loss = criterion(outputs, labels)
            
            test_loss += loss.item()

            # output_prob.append(outputs.cpu().detach().numpy())
            # targets.append(labels.cpu().detach().numpy())
            output_prob[idx] = outputs.cpu().detach().numpy()
            targets[idx] = labels.cpu().detach().numpy()

        output_prob = np.concatenate(output_prob, axis=0)
        targets = np.concatenate(targets, axis=0)

        AUC_macro = roc_auc_score(targets, output_prob, multi_class='ovo', average='macro')

        return model, test_loss, AUC_macro

    @torch.no_grad()
    def get_auc(model, test_loader):

        output_prob = []
        targets = []

        for i, data in enumerate(test_loader):
            model.eval()
            inputs, labels = data['image'], data['target']
            inputs = inputs.to(device).contiguous()
            labels = labels.to(device).contiguous()

            # proba from CNN
            outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)

            output_prob.append(outputs.cpu().detach().numpy())
            targets.append(labels.cpu().detach().numpy())
        
        # concat all mini-batch output
        output_prob = np.concatenate(output_prob, axis=0)
        targets = np.concatenate(targets, axis=0)
        print(output_prob.shape, targets.shape)

        AUC_macro = roc_auc_score(targets, output_prob, multi_class='ovo', average='macro')
        
        return AUC_macro

    # def training_loop(model, criterion, optimizer, train_loader, test_loader, epochs, device, print_step=10):
    import copy
    train_loss_list, test_loss_list, test_auc_list = [], [], []


    best_auc = 0
    best_model = None
    best_epoch = 0

    scaler = GradScaler()

    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple epochs

        model, optimizer, epoch_loss = train(model, criterion, optimizer, train_loader, device, epoch)

        train_loss_list.append(epoch_loss)
        

        with torch.no_grad():
            model, test_loss, test_auc = test(model, criterion, test_loader, device)
            test_loss_list.append(test_loss)
            test_auc_list.append(test_auc)

        scheduler.step()

        # compute ACC
        # test_auc = get_auc(model, test_loader)
        # test_auc_list.append(test_auc)

        # update best model
        if test_auc > best_auc:
            best_auc = test_auc
            best_model = copy.deepcopy(model)
            best_epoch = epoch

            if args.save_model:
                model_path = args.model_dir + '-' + args.task_comment+\
                '-' + pretrain_str + '-' + 'LR=' + str(args.lr) +\
                '-Batch=' + str(args.batch_size) + '-epoch=' + str(args.epochs) +\
                '-' + args.task + '-' + '-model'

                torch.save(best_model.state_dict(), model_path)

        if epoch %args.print_step == 0:
            print(f'Epoch:{epoch}\t'
                f'Train Loss:{epoch_loss}\t'
                f'Val Loss:{test_loss}\t'
                f'Epoch AUC:{test_auc}')
        wandb.log({"AUC": test_auc,
                    "loss": test_loss})
    print('Finished Training in {}, Best AUC is {} on {}th epoch'.format(args.epochs, best_auc, best_epoch))

    # save model
    if args.save_model:
        model_path = args.model_dir + '-' + args.task_comment+\
                '-' + pretrain_str + '-' + 'LR=' + str(args.lr) +\
                '-Batch=' + str(args.batch_size) + '-epoch=' + str(args.epochs) +\
                '-' + args.task + '-' + '-model'

        torch.save(best_model.state_dict(), model_path)

    ## load model state dict
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    return best_model, train_loss_list, test_loss_list, test_auc_list


### test function


@torch.no_grad()
def start_test(model, test_loader):
    print('Start testing model...')
    print('---------------------------')
    model.eval()
    output_prob = []
    targets = []

    for i, data in enumerate(test_loader):
        model.eval()
        inputs, labels = data['image'], data['target']
        inputs = inputs.to(device).contiguous()
        labels = labels.to(device).contiguous()

        # proba from CNN
        outputs = model(inputs)
        # outputs = torch.sigmoid(outputs)

        output_prob.append(outputs.cpu().detach().numpy())
        targets.append(labels.cpu().detach().numpy())
    
    # concat all mini-batch output
    output_prob = np.concatenate(output_prob, axis=0)
    targets = np.concatenate(targets, axis=0)

    # auc_mi_s = roc_auc_score(targets, output_prob, multi_class='ovo',
    #  average='micro') # subclass only allow micro, superclass use macro

    AUC_macro = roc_auc_score(targets, output_prob, multi_class='ovo',
        average='macro')
    
    return AUC_macro
    



def plot_result(train_loss_list, test_loss_list, test_auc_list):
    result = [train_loss_list, test_loss_list, test_auc_list]
    path = args.model_dir + 'train_log.pkl'
    a_file = open(path, "wb")
    pickle.dump(result, a_file)
    a_file.close()

    plt.figure(figsize=(18, 6))
    plt.subplot(3,1,1)
    plt.plot(train_loss_list)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('train loss')
    
    plt.subplot(3,1,2)
    plt.plot(test_loss_list)
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('test loss')

    plt.subplot(3,1,3)
    plt.plot(test_auc_list)
    plt.xlabel('epochs')
    plt.ylabel('AUC')
    plt.title('test auc')

    figpath = args.model_dir + str(args.threshold_ratio) + '-result-curve.png' 
    plt.savefig(figpath, dpi=300)
    print('10 figures plotted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/superclass/shuffle-data')
    parser.add_argument('--task', type=str, default='CXR8')
    parser.add_argument('--model_dir', type=str, default='CXR8/')
    parser.add_argument('--seed', type=int, default=33)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--criterion", default="CEL")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--lr", type=float, default= 0.03)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_dec_rate", type=float, default=0.1)
    parser.add_argument("--lr_dec_step", type=int, default=150)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--print_step', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='resnet')
    parser.add_argument('--task_comment', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    args = parser.parse_args()

    if args.seed:
        setup_seed(args.seed)
    print(os.getcwd())
    os.chdir('/rds/general/user/cl522/home/chest-scan-work/')
    path = args.task + '/'

    if not os.path.exists(path):
        print('create directory {} for save result!'.format(args.task))
        print('---------------------------')
        os.mkdir(path)
    else:
        print('directory {} existing for save result!'.format(args.task))

    if args.pretrain:
        pretrain_str = 'pretrain'
    else:
        pretrain_str = 'no-pretrain'
    if args.model_type:
        wandb_task_name = args.task + '-' + str(args.epochs) + '-' +\
            args.model_type + '-' + pretrain_str + '-HPC_task'
    else:
        wandb_task_name = args.task + '-' + str(args.epochs) + '-HPC_task'

    wandb.init(project="Chest-Scan-baseline", entity="cl522", name=wandb_task_name)

    wandb.config = {
    "task name": args.task,
    "task lr": args.lr,
    "task batch_size": args.batch_size,
    "task criterion": args.criterion,
    "task epochs": args.epochs
    }
    
    args.model_dir = path + '/'  
    train_loader, test_loader = data_prep(args.task, args.batch_size)

    print('Training set Prepared!')
    print('---------------------------')

    # build the model
    if args.model_type == 'vit':
        print('Using no-pretrained ViT model!')
        
        if args.task == 'CXR8':
            
            model = ViT(
                        image_size = 256,
                        patch_size = 32,
                        num_classes = 8,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1,
                        channels=1
                    )
        else:
            
            model = ViT(
                        image_size = 256,
                        patch_size = 32,
                        num_classes = 13,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1,
                        channels=1
                    )
    
    model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[-1])
    model = model.to(device)



    # training
    best_model, train_loss_list, test_loss_list, test_acc_list =\
    start_train(args, model, train_loader, test_loader, device)
    
    # ploting
    # plot_result(train_loss_list, test_loss_list, test_acc_list)

    # testing
    y_pred = start_test(best_model, test_loader)


    
    
    
# python train_temp.py --task=superclass --task_num=1 --epochs=100 --threshold_ratio=0.1

# cd /data/che/LL-work/PTB_XL_data/
# conda activate PTB