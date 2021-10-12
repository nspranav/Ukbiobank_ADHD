#%%
import sys
import numpy as np
from numpy.core.numeric import indices
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from custom_dataset import CustomDataset
from network_classification import Network_classification
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
batch_size = 20
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

model = Network_classification()
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('GPU is available')
    model.cuda()
else:
    print('GPU Not available')

#%%
criterion = nn.NLLLoss(reduction='sum')
optimizer = optim.SGD(model.parameters(),lr=0.003)

scheduler = StepLR(optimizer, step_size=10, gamma=0.3)

epochs = 75

train_losses, validation_losses = [],[]

print('Starting to Train...')

for e in range(epochs):
    model.train()
    train_loss = 0
    for imgs,labels in train_loader:

        if train_on_gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()

        log_ps = model(torch.unsqueeze(imgs,1).float())
        
        #print(log_ps)
        #print(labels.long())

        loss = criterion(log_ps,labels.long())

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    else:
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for imgs,labels in valid_loader:
                if train_on_gpu:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                
                log_ps = model(torch.unsqueeze(imgs,1).float())

                loss = criterion(log_ps,labels.long())
                #print(loss.shape)
                validation_loss += loss.item()

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1,dim=1)
                equals = labels == top_class[:,0]
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(train_loss/len(train_loader))
            validation_losses.append(validation_loss/len(valid_loader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Validation Loss: {:.3f}.. ".format(validation_losses[-1]),
              "Validation accuracy: {:.3f}".format(accuracy/len(valid_loader)))
    
    scheduler.step()
# %%
