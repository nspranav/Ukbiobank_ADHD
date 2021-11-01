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
batch_size = 23
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
                            sampler= train_sampler, num_workers=num_workers)
valid_loader = DataLoader(data,batch_size=batch_size, 
                            sampler= valid_sampler, num_workers=num_workers)
test_loader = DataLoader(data,batch_size = batch_size, 
                            sampler = test_sampler, num_workers=num_workers)
# %%


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network().to(device)


#%%
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.003)


epochs = 50
train_losses, validation_losses = [],[]

print('Starting to Train...')

for e in range(epochs):
    model.train()
    train_loss = 0

    for X,y in train_loader:

        X,y = X.to(device),y.to(device)

        pred = model(torch.unsqueeze(X,1).float())
        loss = criterion(pred,torch.unsqueeze(y,1).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    else:
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X,y in valid_loader:
                X,y = X.to(device),y.to(device)

                pred = model(torch.unsqueeze(X,1).float())
                loss = criterion(pred,torch.unsqueeze(y,1).float())

                valid_loss += loss.item()

            
        print("Epoch {}/{}".format(e+1,epochs),
                "train loss = {:.5f}".format(train_loss/len(train_loader)),
                "validation loss = {:.5f}".format(valid_loss/len(valid_loader)))




    

# %%
