#%%
import numpy as np
from numpy.core.numeric import indices
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from custom_dataset import CustomDataset
from network import Network
import torch
import pickle
import argparse
import os

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

torch.manual_seed(52)
# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 30
# percentage of training set to use as validation
valid_size = 0.20
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network().to(device)


#%%
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.0009)


epochs = 100
train_losses, validation_losses = [],[]


print('Starting to Train...')


for e in range(1,epochs+1):
    model.train()
    train_loss = 0

    actual_train = np.array([])
    actual_valid = np.array([])
    pred_train = np.array([])
    pred_valid = np.array([])

    for X,y in train_loader:

        if e%5 == 0:
            actual_train = np.append(actual_train,y.numpy())

        X,y = X.to(device),y.to(device)

        optimizer.zero_grad()
        

        pred = model(torch.unsqueeze(X,1).float())

        if e%5 == 0:
            pred_train = np.append(pred_train,torch.flatten(pred).detach().to('cpu').numpy())
        
        loss = criterion(pred,torch.unsqueeze(y,1).float())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    else:
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X,y in valid_loader:

                if e%5 == 0:
                   actual_valid = np.append(actual_valid,y.numpy())

                X,y = X.to(device),y.to(device)

                pred = model(torch.unsqueeze(X,1).float())

                if e%5 == 0:
                    pred_valid = np.append(pred_valid,torch.flatten(pred).
                                        detach().to('cpu').numpy())

                loss = criterion(pred,torch.unsqueeze(y,1).float())

                valid_loss += loss.item()

        values = {
            'actual_train' : actual_train,
            'actual_valid' : actual_valid,
            'pred_train' : pred_train,
            'pred_valid' : pred_valid
        }

        if e%5 == 0:
            with open(path +'/arrays'+str(e)+'.pk','wb') as f:
                pickle.dump(values,f)

        print("Epoch {}/{}".format(e,epochs),
                "train loss = {:.5f}".format(train_loss/len(train_loader)),
                "validation loss = {:.5f}".format(valid_loss/len(valid_loader)))

# %%

