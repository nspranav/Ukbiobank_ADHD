#%%
import sys
import numpy as np
from numpy.core.numeric import indices
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from custom_dataset import CustomDataset
from network import Network
import torch
from torch.optim.lr_scheduler import StepLR


#%%

########################
# Loading the Data #####
########################

print(sys.prefix)

# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 15
# percentage of training set to use as validation
valid_size = 0.1
# percentage of data to be used for testset
test_size = 0.05


data = CustomDataset(transform = 
                        transforms.Compose([
                        transforms.ToTensor()]))


# obtaining indices that will be used for train, validation, and test

num_train = len(data)
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

train_loader = DataLoader(data,batch_size=batch_size, 
                            sampler= train_sampler)
valid_loader = DataLoader(data,batch_size=batch_size, 
                            sampler= valid_sampler)
test_loader = DataLoader(data,batch_size = batch_size, 
                            sampler = test_sampler)
# %%

model = Network()
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('GPU is available')
    model.cuda()
else:
    print('GPU Not available')

#%%
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.005)

scheduler = StepLR(optimizer, step_size=20, gamma=0.3)

epochs = 150

train_losses, validation_losses = [],[]

print('Starting to Train...')

pred_train = np.array([])
pred_validation = np.array([])
labels_train = np.array([])
labels_validation = np.array([])

for e in range(epochs):
    model.train()
    train_loss = 0

    # array used for storing predicted values of train and test

    for imgs,labels in train_loader:
        
        labels_train = np.append(labels_train,labels)

        if train_on_gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()

        output = model(torch.unsqueeze(imgs,1).float())

        #values used for plotting
        pred_train = np.append(pred_train,output.cpu().detach()
                                        .numpy().reshape((-1,)))

        loss = criterion(output,torch.unsqueeze(labels,1).float())

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    else:
        validation_loss = 0
        with torch.no_grad():
            model.eval()
            for imgs,labels in valid_loader:

                labels_validation = np.append(labels_validation,labels)

                if train_on_gpu:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                
                pred = model(torch.unsqueeze(imgs,1).float())

                pred_validation = np.append(pred_validation,pred.cpu().detach()
                                        .numpy().reshape((-1,)))

                loss = criterion(pred,torch.unsqueeze(labels,1))

                validation_loss += loss.item()
                
            
            train_losses.append(train_loss/len(train_loader))
            validation_losses.append(validation_loss/len(valid_loader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(validation_losses[-1]))
    
    scheduler.step()

#plotting values
from matplotlib import pyplot as plt

plt.plot(labels_train,pred_train,'.g')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
plt.savefig('reg_train.png')

plt.clf()

plt.plot(labels_validation,pred_validation,'.r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
plt.savefig('reg_valid.png')

# %%
