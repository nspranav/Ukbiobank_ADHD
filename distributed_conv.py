#%%
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from custom_dataset import CustomDataset
from network import Network
import torch
import argparse
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#%%

parser = argparse.ArgumentParser()
parser.add_argument('job_id',type=str)
args = parser.parse_args()
print(args.job_id)
print('number of gpus ',torch.cuda.device_count())

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


torch.manual_seed(52)
# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 25
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

model = Network()

# Loading the model from Job 5436878

load_path = os.path.join(parent_directory,'5436878','models','epoch_28')

model.load_state_dict(torch.load(load_path))

# Freezing the conv layers but not the Batch Norm Layers
for name, param in model.named_parameters():
    if '5' not in name:
        param.requires_grad = False
    if 'bn' in name:
        param.requires_grad = True

model.fc1 = nn.Sequential(nn.Linear(512,256),
                nn.Linear(256,6))


#%%

epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.03)

# adding regularization


#%%

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    batch_size *= torch.cuda.device_count()

model.to(device)



# %%
print('Starting to Train...')


for e in range(1,epochs+1):
    model.train()
    train_loss = 0
    num_correct_train = 0

    actual_train = torch.tensor([]).to(device)
    actual_valid = torch.tensor([]).to(device)
    pred_train = torch.tensor([]).to(device)
    pred_valid = torch.tensor([]).to(device)

    for X,y in train_loader:
        
        X,y = X.to(device),y.to(device)

        actual_train = torch.cat((actual_train,y),0)

        optimizer.zero_grad()

        pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))

        pred_train = torch.cat((pred_train,torch.max(F.softmax(pred,dim=1), dim=1)[1]),0)
        
        loss = criterion(pred,y)

        loss.backward()
        optimizer.step()
        #print('loss =',loss.item())
        train_loss += loss.item()
        correct = torch.eq(torch.max(F.softmax(pred,dim=1), dim=1)[1],y).view(-1)
        num_correct_train += torch.sum(correct).item()
    else:
        model.eval()
        valid_loss = 0
        num_correct_valid = 0

        with torch.no_grad():
            for X,y in valid_loader:

                X,y = X.to(device),y.to(device)

                actual_valid = torch.cat((actual_valid,y),0)

                pred = model(torch.unsqueeze(X,1).float())

                

                loss = criterion(pred,y.long())

                valid_loss += loss.item()
                correct = torch.eq(torch.max(F.softmax(pred,dim=1), dim=1)[1],y).view(-1)
                pred_valid = torch.cat((pred_valid,torch.max(F.softmax(pred,dim=1), dim=1)[1]),0)
                num_correct_valid += torch.sum(correct).item()

        # values = {
        #     'actual_train' : actual_train,
        #     'actual_valid' : actual_valid,
        #     'pred_train' : pred_train,
        #     'pred_valid' : pred_valid
        # }

        # if e % 5 == 0:
        #     with open(path + '/arrays'+str(e)+'.pk', 'wb') as f:
        #         pickle.dump(values, f)
        
        #compute the r square

        
        plt.figure()
        plt.plot(actual_train.detach().cpu().numpy(),pred_train.detach().cpu()
                .numpy(),'.')
        plt.title('Train - True vs pred')
        plt.xlabel('True numeric_score')
        plt.ylabel('Predicted numeric_score')
        
        writer.add_figure('Train - True vs pred', plt.gcf(),e,True)
        

        plt.figure()
        plt.plot(actual_valid.detach().cpu().numpy(),pred_valid.detach().cpu()
                .numpy(),'.')
        plt.title('Validation - True vs pred')
        plt.xlabel('True score')
        plt.ylabel('Predicted score')
        
        writer.add_figure('Validation - True vs pred', plt.gcf(),e,True)

        print("Epoch: {}/{}.. ".format(e, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
              'Train Accuracy: {:.3f}..'.format(num_correct_train/len(train_idx)),
              "validation Accuracy: {:.3f}..".format(num_correct_valid/len(valid_idx))
            )
              

        writer.add_scalar('Train Loss', train_loss/len(train_loader),e)
        writer.add_scalar('Validation Loss', valid_loss/len(valid_loader),e)
        writer.add_scalar('Train Accuracy',num_correct_train/len(train_idx),e)
        writer.add_scalar('validation Accuracy', num_correct_valid/len(valid_idx),e)
        if abs(valid_loss/len(valid_loader) - train_loss/len(train_loader)) < 0.2:
           torch.save(model.state_dict(), os.path.join(model_save_path,
                'epoch_'+str(e)))

writer.flush()
writer.close()

# %%

