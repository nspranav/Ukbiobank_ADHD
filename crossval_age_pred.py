#%%
import numpy as np
import pandas as pd
import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

from custom_dataset import CustomDataset
from network import Network
from utils import *

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

#%%
torch.set_default_dtype(torch.float32)
parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())

#creating directory

directory = args.job_id
parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs'
path = os.path.join(parent_directory,directory)
model_save_path = os.path.join(path,'models_fold')

if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(model_save_path)



#%%

########################
# Loading the Data #####
########################



torch.manual_seed(52)
# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 15
if torch.cuda.device_count() > 1:
    batch_size *= torch.cuda.device_count()
else:
    batch_size = 5
# percentage of training set to use as validation
valid_size = 0.20
# percentage of data to be used for testset
test_size = 0.10


train_data = CustomDataset(transform = 
                        transforms.Compose([
                            transforms.RandomHorizontalFlip()
                            ]),train=True)

valid_data = CustomDataset(train=False,valid=True)

test_data = CustomDataset(train=False,valid=False)

# get filtered variables
vars = valid_data.vars

#%% 

# Prepare for k-fold

sss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=52)
learning_rate = 0.0001
fold = 1

for train_idx, valid_idx in sss.split(np.zeros_like(vars),vars.new_score.values):
    writer = SummaryWriter(log_dir=path+'/fold'+str(fold))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_data.test_idx)

    train_loader = DataLoader(train_data,batch_size=batch_size, 
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_data,batch_size=batch_size, 
                                sampler= valid_sampler, num_workers=num_workers)

    test_loader = DataLoader(test_data,batch_size=batch_size,
                                sampler= test_sampler, num_workers=num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = Network()
    model.fc1 = nn.Sequential(nn.Linear(512,2))

    # model.fc1 = nn.Sequential(nn.Linear(512,1))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    load_path = os.path.join(parent_directory,'1504485','models_fold','2','epoch_45')

    model.load_state_dict(torch.load(load_path))
    
    # model = model.module

    #model = nn.DataParallel(model)

    for name, param in model.named_parameters():
        if 'fc1' in name or '5' in name or '4' in name or '3' in name:
            print(name)
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(model)


    #%%

    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.0001)

    #%%

    model.to(device)

    print('Starting to Train...')

    for e in range(1,epochs+1):
        model.train()
        train_loss = 0
        num_correct_train = 0

        actual_train = torch.tensor([]).to(device)
        actual_valid = torch.tensor([]).to(device)
        actual_test = torch.tensor([]).to(device)
        pred_proba_train = torch.tensor([]).to(device)
        pred_train = torch.tensor([]).to(device)
        pred_valid = torch.tensor([]).to(device)
        pred_proba_valid = torch.tensor([]).to(device)
        pred_test = torch.tensor([]).to(device)
        pred_proba_test = torch.tensor([]).to(device)

        for X,y in train_loader:

            X,y = X.to(device),y.to(device)
            #print(actual_train.dtype,y.dtype,actual_train.shape,y.shape)
            actual_train = torch.cat((actual_train,y),0)

            optimizer.zero_grad()

            pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
            if(torch.sum(torch.isnan(pred))>0):
                print(f'{e},{pred}')
                exit(-1)
            soft_max = F.softmax(pred,dim=1)
            pred_train = torch.cat((pred_train,torch.max(soft_max, dim=1)[1]),0)
            loss = criterion(pred,y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
            pred_proba_train = torch.cat((pred_proba_train,soft_max[:,-1]),0)
            num_correct_train += torch.sum(correct).item()
        else:
            model.eval()
            valid_loss = 0
            num_correct_valid = 0

            with torch.no_grad():
                for X,y in valid_loader:

                    X,y = X.to(device),y.to(device)

                    actual_valid = torch.cat((actual_valid,y),0)
                    pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
                    try:
                        loss = criterion(pred,y)
                    except:
                        print(pred)
                        print(y)

                    valid_loss += loss.item()
                    soft_max = F.softmax(pred,dim=1)
                    pred_proba_valid = torch.cat((pred_proba_valid,soft_max[:,-1]),0)
                    correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
                    pred_valid = torch.cat((pred_valid,torch.max(soft_max, dim=1)[1]),0)
                    num_correct_valid += torch.sum(correct).item()
        model.eval()
        test_loss = 0
        num_correct_test = 0

        with torch.no_grad():
            for X,y in test_loader:

                X,y = X.to(device),y.to(device)

                actual_test = torch.cat((actual_test,y),0)
                pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
                try:
                    loss = criterion(pred,y)
                except:
                    print(pred)
                    print(y)

                test_loss += loss.item()
                soft_max = F.softmax(pred,dim=1)
                pred_proba_test = torch.cat((pred_proba_test,soft_max[:,-1]),0)
                correct = torch.eq(torch.max(soft_max, dim=1)[1],y).view(-1)
                pred_test = torch.cat((pred_test,torch.max(soft_max, dim=1)[1]),0)
                num_correct_test += torch.sum(correct).item()

        mcc_t,f1_t,b_a_t = write_confusion_matrix(writer, actual_train.detach().cpu().numpy(),
            pred_train.detach().cpu().numpy(), e,'Confusion Matrix - Train' )
        mcc_v,f1_v,b_a_v = write_confusion_matrix(writer,actual_valid.detach().cpu().numpy(),
            pred_valid.detach().cpu().numpy(), e,'Confusion Matrix - Validation')
        mcc_test,f1_test,b_a_test = write_confusion_matrix(writer,actual_test.detach().cpu().numpy(),
            pred_test.detach().cpu().numpy(), e,'Confusion Matrix - Validation')
        
        auc_train = roc_auc_score(actual_train.detach().cpu().numpy(),pred_proba_train.detach().cpu().numpy())
        auc_valid = roc_auc_score(actual_valid.detach().cpu().numpy(),pred_proba_valid.detach().cpu().numpy())
        auc_test = roc_auc_score(actual_test.detach().cpu().numpy(),pred_proba_test.detach().cpu().numpy())  
        
        print("Epoch: {}/{}.. ".format(e, epochs),
            "Training Accuracy: {:.3f}.. ".format(num_correct_train/len(train_idx)),
            "Train Bal acc: {:.4f}.. ".format(b_a_t),
            "AUC train: {:4f}.. ".format(auc_train),
            "Validation Accuracy: {:.3f}.. ".format(num_correct_valid/len(valid_idx)),
            "Val Bal acc: {:.4f}.. ".format(b_a_v),
            "AUC valid: {:4f}.. ".format(auc_valid),
            "Test Accuracy: {:.3f}.. ".format(num_correct_test/len(test_data.test_idx)),
            "Test Bal acc: {:.4f}.. ".format(b_a_test),
            "AUC Test: {:4f}.. ".format(auc_test)
        )
            
        # writer.add_scalar('Train r2', r2_score(pred_train,actual_train),e)
        # writer.add_scalar('Valid r2', r2_score(pred_valid,actual_valid),e)
        writer.add_scalar('Train Loss', train_loss/len(train_loader),e)
        writer.add_scalar('Validation Loss', valid_loss/len(valid_loader),e)
        writer.add_scalar('Train Accuracy',num_correct_train/len(train_idx),e)
        writer.add_scalar('validation Accuracy', num_correct_valid/len(valid_idx),e)
        
        if abs(valid_loss/len(valid_loader) - train_loss/len(train_loader)) < 0.2:
            fold_path = os.path.join(model_save_path,str(fold))

            if not os.path.exists(fold_path):
                os.makedirs(fold_path)
            torch.save(model.state_dict(), os.path.join(fold_path,
                'epoch_'+str(e)))

    fold+=1
    print('####################################################################')
    writer.flush()
    writer.close()

# %%

