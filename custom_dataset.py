#%%
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
from torchvision import transforms
import random
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit
#%%

class CustomDataset(Dataset):
    
    def __init__(self,img_path= '/data/qneuromark/Data/UKBiobank/Data_BIDS/Raw_Data'
                    , label_file= 'subset_vars.csv',transform=None
                    , target_transform=None,train=True,valid=False,random_state=52):
        path = '/data/users2/pnadigapusuresh1/Projects/ukbiobank/Data/'
        self.img_path = img_path
        self.dirs = os.listdir(img_path)
        # The below list contains the index of directories that are missing the
        # mri images. 
        self.no_smri = [8073,12877,15213,18350,29723] #indices
        #[1336082,2464107, 4158450, 3429974, 4468630] subject ids
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
                            'sex_f31_0_0','age_when_attended_assessment_centre_f21003_0_0'])
        self.vars.columns = ['sex','score','age']

        #self.vars['score'] = self.vars['score'] + 1 
        
        # Applying log transform
        #self.vars = self.vars.apply(np.log,axis=1)

        #self.vars['score'] = SimpleImputer(strategy='mean',
        #                       missing_values=np.nan).fit_transform(self.vars)
        
        # removing missing scores 
        self.vars = self.vars.loc[
                        self.vars.score.isin([2,3,4,5,9,10,11,12])]

        #######
        self.vars['new_score'] = [0 if a < 6 else 1 for a in self.vars['score']]

        # We want sampling only during the training and validation and not for
        # testing the fixed model
        
        # if not test:
        #     maj_class = resample(self.vars[self.vars.new_score == 0],
        #             n_samples = 2250,replace=False,random_state=random_state)
        #     min_class = self.vars[self.vars.new_score == 1]
        #     self.vars = pd.concat([maj_class,min_class])

        # Need to sort by index because we want the order of data same for
        # both the train and validation dataset
        #self.vars.drop([1336082,2464107, 4158450, 3429974, 4468630],inplace=True)
                
        #######

        # Removing the images with no pixels
        self.vars = self.vars.drop([1171080,1660210,2012720,2378544,2835040,2951207,4312676],axis=0)
        
        self.vars['pos'] = list(range(5462))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        train_idx, self.test_idx = next(sss.split(np.zeros_like(self.vars),
            self.vars.new_score.values))
        if train or valid:
            self.vars = self.vars.iloc[train_idx]
        else:
            test_vars = self.vars.iloc[self.test_idx]
            self.female_idx = test_vars[test_vars.sex == 0].pos.tolist()
            self.male_idx = test_vars[test_vars.sex==1].pos.tolist()
        self.vars = self.vars.sort_index()    
        self.dirs = self.vars.index

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    
    def __len__(self):
        return len(self.dirs)

    def __getitem__(self,idx):
        try:
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_01/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        except OSError:
            img = nib.load(os.path.join(self.img_path,str(self.dirs[idx])
                            ,'ses_02/anat/Sm6mwc1pT1.nii.nii')).get_fdata()
        label = self.vars.iloc[idx]

        ########################################

        #transforms to be done on every image with probability of 0.5

        
        img = torch.tensor(img)
        img = (img - img.mean())/img.std()
        if(torch.sum(torch.isnan(img))> 0):
            print(f'custom dataset,{idx}')
            exit(-1)
        if self.train:
            # p = random.random()
            # if p > 0.5:
            #     t = transforms.Pad((1,1,1,1),0)
                
            #     choice = random.choice([-2,-1,1,2])

            #     if choice == -2:
            #         img = t(img)
            #         img = t(img)

            #         img = img[:,:-4,:-4]

            #         tmp = torch.zeros((2,145,121))

            #         img = torch.cat((img,tmp),0)

            #         img = img[2:,:,:]


            #     elif choice == -1:
            #         img = t(img)
            #         tmp = torch.zeros((2,145,121))
            #         img = img[:,:-2,:-2]

            #         tmp = torch.zeros((1,145,121))
            #         img = torch.cat((img,tmp),0)
            #         img = img[1:,:,:]
                
            #     elif choice == 1:
            #         img = t(img)

            #         img = img[:,2:,2:]

            #         tmp = torch.zeros((1,145,121))
            #         img = torch.cat((tmp,img),0)
            #         img = img[:-1,:,:]
                
            #     elif choice == 2:
            #         img = t(img)
            #         img = t(img)

            #         img = img[:,4:,4:]

            #         tmp = torch.zeros((2,145,121))
            #         img = torch.cat((img,tmp),0)
            #         img = img[:-2,:,:]
            
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                label = self.target_transform(label)

        #offset by 4 because of scores range from 4 to 9
        return img,int(label['new_score']),label['age']
# %%



# %%
