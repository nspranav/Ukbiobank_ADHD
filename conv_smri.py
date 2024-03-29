#%%
import os
import argparse
import numpy as np
from numpy.core.numeric import indices
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from custom_dataset import CustomDataset
from network import Network
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


#%%


parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args.job_id)


#creating directory

directory = args.job_id
parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs'
path = os.path.join(parent_directory,directory)
model_save_path = os.path.join(path,'models')

if not os.path.exists(path):
    os.mkdir(path)
    os.mkdir(model_save_path)


#%%
########################
# Loading the Data #####
########################

writer = SummaryWriter(log_dir=path)

# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 25
# percentage of training set to use as validation
valid_size = 0.1
# percentage of data to be used for testset
test_size = 0.05

valid_data = CustomDataset(train=False,valid=True)

test_data = CustomDataset(train=False,valid=False)

# obtaining indices that will be used for train, validation, and test

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network()

# Loading the model from Job 5436878

load_path = os.path.join(parent_directory,'5436878','models','epoch_28')

model.load_state_dict(torch.load(load_path))

# Freezing the conv layers but not the Batch Norm Layers
for name, param in model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

model.fc1 = nn.Sequential(nn.Dropout(),nn.Linear(512,2))

model = model.to(device)

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

#scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

epochs = 100

train_losses, validation_losses = [],[]

print('Starting to Train...')

# %%

for e in range(epochs):
    model.train()
    train_loss = 0
    num_correct_train = 0

    for X,y in train_loader:
        
        #labels_train = np.append(labels_train,labels)

        X,y = X.to(device),y.to(device)
    
        optimizer.zero_grad()
        pred = model(torch.unsqueeze(X,1).float())

        loss = criterion(pred,y.long())
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct = torch.eq(torch.max(F.softmax(pred,dim=1), dim=1)[1],
							   y).view(-1)
        num_correct_train += torch.sum(correct).item()

    else:
        validation_loss = 0
        num_correct_valid = 0

        model.eval()
        with torch.no_grad():
            for X,y in valid_loader:


                X,y = X.to(device),y.to(device)
                
                pred = model(torch.unsqueeze(X,1).float())

                loss = criterion(pred,y.long())

                validation_loss += loss.item()
                correct = torch.eq(torch.max(F.softmax(pred,dim=1), dim=1)[1],
							   y).view(-1)
                num_correct_valid += torch.sum(correct).item()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Test Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
              'Train Accuracy: {:.3f}..'.format(num_correct_train/len(train_idx)),
              "validation Accuracy: {:.3f}..".format(num_correct_valid/len(valid_idx)))
            
            writer.add_scalar('Train Loss', train_loss/len(train_loader),e)
            writer.add_scalar('Validation Loss', validation_loss/len(valid_loader),e)
            writer.add_scalar('Train Accuracy',num_correct_train/len(train_idx),e)
            writer.add_scalar('validation Accuracy', num_correct_valid/len(valid_idx),e)
    
    #scheduler.step(validation_losses[-1])

#plotting values
# from matplotlib import pyplot as plt

# plt.plot(labels_train,pred_train,'.g')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.show()
# plt.savefig('reg_train.png')

# plt.clf()

# plt.plot(labels_validation,pred_validation,'.r')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.show()
# plt.savefig('reg_valid.png')

# %%
