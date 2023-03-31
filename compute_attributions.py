#%%
import numpy as np
import pandas as pd
import os
import json
from captum.attr import IntegratedGradients,Occlusion
from nilearn.image import load_img,resample_img 

import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from custom_dataset import CustomDataset
from network import Network
from utils import *
import torch.nn.functional as F

#%%
parent_directory = '/data/users2/pnadigapusuresh1/JobOutputs'
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Network()
model.fc1 = nn.Sequential(nn.Linear(512,2))
model = nn.DataParallel(model)
model.to(device)

# Loading the model from Job 5436878
#loading model from 6066159
#load_path = os.path.join(parent_directory,'6066159','models','epoch_170')

#loading model from 1780660
load_path = os.path.join(parent_directory,'1818979','models_fold','5','epoch_38')
model.load_state_dict(torch.load(load_path))
model.eval()

torch.manual_seed(52)
np.random.seed(52)
# number of subprocesses to use for data loading
num_workers = 1
# how many samples per batch to load
batch_size = 1

valid_data = CustomDataset(train= False,valid=False)

# get filtered variables
vars = valid_data.vars.iloc[valid_data.test_idx]

valid_sampler = SubsetRandomSampler(valid_data.male_idx)

valid_loader = DataLoader(valid_data,batch_size=batch_size, 
                            sampler= valid_sampler, num_workers=num_workers)

X_all = np.zeros((121,145,121))
for X,y,age in valid_loader:
    X_all = np.add(X_all , X.squeeze())
X_all /= len(valid_loader)
X_all = np.expand_dims(np.expand_dims(X_all,axis =0),axis=0)
X_all = torch.tensor(X_all).float().to(device)

ig = Occlusion(model)

attr_0 = attr_1 = np.zeros((121,145,121),dtype = np.float64)

with open('region_labels.json','r') as f:
    l = json.load(f)

#%%
imf = load_img('/trdapps/linux-x86_64/matlab/toolboxes/spm12/tpm/labels_Neuromorphometrics.nii')
aal = load_img('/data/users2/pnadigapusuresh1/Downloads/AAL3/AAL3v1.nii.gz')
aal_resampled = resample_img('/data/users2/pnadigapusuresh1/Downloads/AAL3/aal.nii.gz',target_affine=imf.affine,target_shape=imf.shape).get_fdata()
df = pd.read_csv('/data/users2/pnadigapusuresh1/Downloads/AAL3/aal.nii.txt',sep=' ',index_col=0,header=None,usecols=[0,1],names=['value','regions'])
l = df.to_dict()['regions']
labels = {v:{'attr_sum':0,'attrs':[]} for k,v in l.items()}
#%%
num_0 = num_1 =  0
actual_test = torch.tensor([]).to(device)
deltas = torch.tensor([]).to(device)
pred_test = torch.tensor([]).to(device)
region_means = []
for X,y,age in valid_loader:
    X,y = X.to(device),y.to(device)
    actual_test = torch.cat((actual_test,y),0)
    X.requires_grad_()
    pred = torch.squeeze(model(torch.unsqueeze(X,1).float()))
    soft_max = F.softmax(pred,dim=0)
    pred_test = torch.cat((pred_test,soft_max.argmax().unsqueeze(0)),0)
    if soft_max.argmax() == y:
        # attr, delta = ig.attribute(torch.unsqueeze(X,1).float(), baselines=X_all,target=y, return_convergence_delta=True, 
        #                 internal_batch_size=5,n_steps=550)
        attr = ig.attribute(torch.unsqueeze(X,1).float(),baselines=X_all,target=y,
                sliding_window_shapes=(1,3,3,3))
        #deltas = torch.cat((deltas,delta.unsqueeze(0).float()),0)
        attr = attr.detach().cpu().numpy().astype(dtype=np.float64)
        attr = attr.squeeze()
        #mask = np.abs(attr) > 1e-5 #check this part
        X = X.squeeze()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                        #print(i,j,k)
                        r = aal_resampled[i,j,k]
                        try:
                            labels[l[int(r)]]['attr_sum'] += attr[i,j,k]
                            labels[l[int(r)]]['attrs'].append(attr[i,j,k])
                        except KeyError:
                            pass
        region_means.append({k:v['attr_sum']/len(v['attrs']) for k,v in labels.items()})
        region_means[-1]['memory'] = y.item()
        region_means[-1]['age'] = age.item()
        if y == 0:
            num_0 += 1
            attr_0 += attr
        else:
            num_1 += 1
            attr_1 += attr

male_df = pd.DataFrame.from_dict(region_means)
male_df.to_csv('male_df.csv',index=False)

#%%
with open('attr_mean_male_0.npy', 'wb') as f:
    np.save(f, attr_0)
    print(f'num_0_male {num_0}') #f169 #m147
with open('attr_mean_male_1.npy', 'wb') as f:
    np.save(f, attr_1)
    print(f'num_1_male {num_1}') #f90 #m65