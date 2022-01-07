#%%
import argparse
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from custom_dataset import CustomDataset

class Network(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.cv1 = nn.Conv3d(1, 32, 3, stride=1, padding=0) 
        self.bn1 = nn.BatchNorm3d(32)
        self.cv2 = nn.Conv3d(32, 64, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.cv3 = nn.Conv3d(64, 128, 3,stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(128)
        self.cv4 = nn.Conv3d(128, 256, 3,stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(256)
        self.cv5 = nn.Conv3d(256, 512, 3,stride=1, padding=0)
        self.bn5 = nn.BatchNorm3d(512)
        self.cv6 = nn.Conv3d(128, 128, 1,stride=1, padding=0)
        self.bn6 = nn.BatchNorm3d(128)
        self.cv7 = nn.Conv3d(64,64,1) 
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,16)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(16,1)
        self.dropout = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.1)
        self.avgpool = nn.AvgPool3d(3)

        self.layer1 = nn.Sequential(self.cv1,self.pool,nn.ReLU(),self.bn1)
        self.layer2 = nn.Sequential(self.cv2,self.pool,nn.ReLU(),self.bn2)
        self.layer3 = nn.Sequential(self.cv3,self.pool,nn.ReLU(),self.bn3)
        self.layer4 = nn.Sequential(self.cv4,self.pool,nn.ReLU(),self.bn4)
        self.layer5 = nn.Sequential(self.cv5,self.pool,nn.ReLU(),self.bn5)
    
    def forward(self,img):
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        img = self.layer5(img)

        img = img.view(img.shape[0], -1)
        img = self.fc1(img)
        img = self.fc2(img)

        return img
    
    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(),lr=0.003)
        return optimizer

    def training_step(self,batch,batch_idx):

        X, y = batch
        pred = self(torch.unsqueeze(X,1).float())
        loss = F.mse_loss(pred,torch.unsqueeze(y,1).float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        pred = self(torch.unsqueeze(X,1).float())
        loss = F.mse_loss(pred,torch.unsqueeze(y,1).float())
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    

#%%

parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args.job_id)

#creating directory

directory = args.job_id
parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs'
path = os.path.join(parent_directory,directory)

if not os.path.exists(path):
    os.mkdir(path)




#%%

########################
# Loading the Data #####
########################


# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 30
# percentage of training set to use as validation
valid_size = 0.50
# percentage of data to be used for testset
test_size = 0.10


train_data = CustomDataset(transform = 
                        transforms.Compose([
                            transforms.RandomHorizontalFlip()
                            ]),train = True)

valid_data = CustomDataset(train = False)

# obtaining indices that will be used for train, validation, and test

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
test_split = int(np.floor(test_size * num_train))
test_idx, train_idx = indices[: test_split], indices[test_split : ]

train_rem = len(train_idx)
valid_spilt = int(np.floor(valid_size * train_rem))

valid_idx, train_idx = indices[: valid_spilt], indices[valid_spilt : ]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(train_data,batch_size=batch_size, 
                            sampler= train_sampler, num_workers=num_workers)
valid_loader = DataLoader(valid_data,batch_size=batch_size, 
                            sampler= valid_sampler, num_workers=num_workers)
test_loader = DataLoader(valid_data,batch_size = batch_size, 
                            sampler = test_sampler, num_workers=num_workers)

                            
# %%
