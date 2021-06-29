#%%
import nibabel as nib
import numpy as np
import os
import pickle
import pandas as pd

# %%
path = '/data/collaboration/NeuroMark2/Data/UKBiobank/Data_BIDS/Raw_Data'
# %%
dirs = os.listdir(path)
no_smri = [2724,11532,19200,24947,27444]    
# %%
# %%
lst = []
not_found = {}
for i,dir in enumerate(dirs):
    try:
        lst.append(nib.load(os.path.join(path,dir,'ses_01/anat/Sm6mwc1pT1.nii.nii')))
    except OSError:
        not_found[i] = dir

# %%
ses02_missing = {}
for key,val in not_found.items():
    try:
        lst.insert(key,nib.load(os.path.join(path,val,'ses_02/anat/Sm6mwc1pT1.nii.nii')))
    except OSError:
       ses02_missing[key] = val
# %%
save_path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/'
filename = 'img_all.npy'

with open(save_path+filename,'rb') as f:
    img_all = np.load(f)

# %%
var_path = '/data/collaboration/NeuroMark2/Data/UKBiobank/Data_info/New_rda/'
var_file = 'subset_vars.csv'
subset_vars = pd.read_csv(save_path+var_file)

# %%
