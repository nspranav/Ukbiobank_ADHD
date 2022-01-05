#%%
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms
import random

#%%

class CustomDataset(Dataset):
    
    def __init__(self,img_path= '/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data'
                    ,label_file= 'subset_vars.csv',transform=None, target_transform=None,train=True):
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
                            usecols=['eid','age_when_attended_assessment_centre_f21003_0_0'])
        self.vars.columns = ['neuroticism_score']
        #self.vars['neuroticism_score'] = self.vars['neuroticism_score'] + 1 
        
        # Applying log transform
        #self.vars = self.vars.apply(np.log,axis=1)

        #self.vars['neuroticism_score'] = SimpleImputer(strategy='mean',
        #                       missing_values=np.nan).fit_transform(self.vars)
        

        self.misssing_scores = self.vars[self.vars['neuroticism_score']
                                        .isnull()].index

        
        self.dirs = list(set(self.dirs) - set(self.misssing_scores) )

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    
    def __len__(self):
        return 8000

    def __getitem__(self,idx):
        try:
            ses2 = False
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_01/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        except OSError:
            ses2 = True
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_02/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        label = self.vars.loc[self.dirs[idx]].values[0]

        ########################################
        #transforms to be done on every image with probability of 0.5

        if self.train:
            img = torch.tensor(img)

            p = random.random()
            if p > 0.5:
                t = transforms.Pad((1,1,1,1),0)
                
                choice = random.choice([-2,-1,1,2])

                if choice == -2:
                    img = t(img)
                    img = t(img)

                    img = img[:,:-4,:-4]

                    tmp = torch.zeros((2,145,121))

                    img = torch.cat((img,tmp),0)

                    img = img[2:,:,:]


                elif choice == -1:
                    img = t(img)
                    tmp = torch.zeros((2,145,121))
                    img = img[:,:-2,:-2]

                    tmp = torch.zeros((1,145,121))
                    img = torch.cat((img,tmp),0)
                    img = img[1:,:,:]
                
                elif choice == 1:
                    img = t(img)

                    img = img[:,2:,2:]

                    tmp = torch.zeros((1,145,121))
                    img = torch.cat((tmp,img),0)
                    img = img[:-1,:,:]
                
                elif choice == 2:
                    img = t(img)
                    img = t(img)

                    img = img[:,4:,4:]

                    tmp = torch.zeros((2,145,121))
                    img = torch.cat((img,tmp),0)
                    img = img[:-2,:,:]
 

            #########################################
            
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

    
        return img,label
# %%



# %%
