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
                    , label_file= 'subset_vars.csv',transform=None
                    , target_transform=None,train=True):
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
        """
        # column for age: age_when_attended_assessment_centre_f21003_0_0
        # column for sex; sex_f31_0_0 
        # column for Numeric memory: 'maximum_digits_remembered_correctly_f4282_2_0'
        """
        self.dirs = np.array(self.dirs,dtype=int)
        self.vars = pd.read_csv(path+label_file,index_col='eid',
                            usecols=['eid','maximum_digits_remembered_correctly_f4282_2_0',
                            'sex_f31_0_0'])
        self.vars.columns = ['sex','score']

        #self.vars['score'] = self.vars['score'] + 1 
        
        # Applying log transform
        #self.vars = self.vars.apply(np.log,axis=1)

        #self.vars['score'] = SimpleImputer(strategy='mean',
        #                       missing_values=np.nan).fit_transform(self.vars)
        
        # removing missing scores 
        self.misssing_scores = self.vars.loc[
                        self.vars.score.isin([-1,6,7,8,np.NaN])].index

        
        self.dirs = list(set(self.dirs) - set(self.misssing_scores) )

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self,idx):
        try:
            ses2 = False
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_01/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        except OSError:
            ses2 = True
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_02/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        label = self.vars.loc[self.dirs[idx]]

        ########################################
        # Binning the values (2,3,4) and (9,10,11,12)

        if label['score'] <= 6:
            label['score'] = 0.0
        # elif label['score'] >= 8:
        #     label['score'] = 2.0
        else:
            label['score'] = 1.0

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

        #offset by 4 because of scores range from 4 to 9
        return img,int(label['score'])
# %%



# %%
