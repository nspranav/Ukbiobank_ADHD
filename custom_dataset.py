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
from sklearn.preprocessing import StandardScaler

#%%

class CustomDataset(Dataset):
    
    def __init__(self,img_path= '/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data'
                    ,label_file= 'subset_vars.csv',transform=None, target_transform=None):
        path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/'
        self.img_path = img_path
        self.dirs = os.listdir(img_path)
        # The below list contains the index of directories that are missing the
        # mri images. 
        self.no_smri = [8073,12877,15213,18350,29723]
        self.dirs.pop(8073)
        self.dirs.pop(12876)
        self.dirs.pop(15211)
        self.dirs.pop(18347)
        self.dirs.pop(29719)

        self.dirs = np.array(self.dirs,dtype=np.int)
        self.vars = pd.read_csv(path+label_file,index_col='eid',
                            usecols=['eid','neuroticism_score_f20127_0_0'])
        self.vars.columns = ['neuroticism_score']
        #self.vars['neuroticism_score'] = SimpleImputer(strategy='mean',
        #                       missing_values=np.nan).fit_transform(self.vars)
        
        #computing the missing scores 
        self.misssing_scores = self.vars[self.vars['neuroticism_score']
                                        .isnull()].index
        self.dirs = list(set(self.dirs) - set(self.misssing_scores) )

        self.transform = transform
        self.target_transform = target_transform

    
    def __len__(self):
        return 8000

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
