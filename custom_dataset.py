#%%
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nibabel as nib
from torchvision import transforms
from sklearn.impute import SimpleImputer

#%%

class CustomDataset(Dataset):
    
    def __init__(self,img_path= '/data/collaboration/NeuroMark2/Data/UKBiobank/Data_BIDS/Raw_Data'
                    ,label_file= 'subset_vars.csv',transform=None, target_transform=None):
        path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/'
        self.img_path = img_path
        self.dirs = os.listdir(img_path)
        self.no_smri = [2724,11532,19200,24947,27444]
        self.dirs.pop(2724)
        self.dirs.pop(11531)
        self.dirs.pop(19198)
        self.dirs.pop(24944)
        self.dirs.pop(27440)
        self.dirs = np.array(self.dirs,dtype=np.int)
        self.vars = pd.read_csv(path+label_file,index_col='eid',
                            usecols=['eid','neuroticism_score_f20127_0_0'])
        self.vars.columns = ['neuroticism_score']
        self.vars['neuroticism_score'] = SimpleImputer(strategy='mean',
                                missing_values=np.nan).fit_transform(self.vars)
        self.transform = transform
        self.target_transform = target_transform
    
    
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self,idx):
        try:
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_01/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        except OSError:
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_02/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        label = self.vars.loc[self.dirs[idx]].values[0]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
    
        return img,label
# %%



# %%
