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

#%%

########################
# Loading the Data #####
########################


# number of subprocesses to use for data loading
num_workers = 4
# how many samples per batch to load
batch_size = 2
# percentage of training set to use as validation
valid_size = 0.1
# percentage of data to be used for testset
test_size = 0.05


data = CustomDataset(transform = 
                        transforms.Compose([
                            transforms.RandomHorizontalFlip(),
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
                            sampler= train_sampler, num_workers=num_workers)
valid_loader = DataLoader(data,batch_size=batch_size, 
                            sampler= valid_sampler, num_workers=num_workers)
test_loader = DataLoader(data,batch_size = batch_size, 
                            sampler = test_sampler, num_workers=num_workers)
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
optimizer = optim.SGD(model.parameters(),lr=0.0001)

epochs = 50
train_losses, validation_losses = [],[]

print('Starting to Train...')

for e in range(epochs):

    train_loss = 0
    for imgs,labels in train_loader:

        if train_on_gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()

        output = model(torch.unsqueeze(imgs,1).float())

        loss = criterion(output,torch.unsqueeze(labels,1).float())

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    else:
        validation_loss = 0
        with torch.no_grad():
            model.eval()
            for imgs,labels in valid_loader:
                if train_on_gpu:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                
                pred = model(torch.unsqueeze(imgs,1).float())

                loss = criterion(pred,torch.unsqueeze(labels,1))

                validation_loss += loss.item()
                
            model.train()
            
            train_losses.append(train_loss/len(train_loader))
            validation_losses.append(validation_loss/len(valid_loader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(validation_losses[-1]))

torch.save({
    'epoch': 10,
    'model_state_dict' : model.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'loss' : loss
},'model.pth')
# %%
