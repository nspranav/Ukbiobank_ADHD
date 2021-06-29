import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv3d(1,32,3,padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.cv2 = nn.Conv3d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.cv3 = nn.Conv3d(64,128,3,padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.cv4 = nn.Conv3d(128,256,3,padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.cv5 = nn.Conv3d(256,256,3,padding=1)
        self.bn5 = nn.BatchNorm3d(256)
        self.cv6 = nn.Conv3d(256,64,1,padding=1)
        self.bn6 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(64*5*5*6,500)
        self.fc2 = nn.Linear(500,250)
        self.fc3 = nn.Linear(250,1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,img):
        img = self.pool(F.relu(self.bn1(self.cv1(img))))
        img = self.pool(F.relu(self.bn2(self.cv2(img))))
        img = self.pool(F.relu(self.bn3(self.cv3(img))))
        img = self.pool(F.relu(self.bn4(self.cv4(img))))
        img = self.pool(F.relu(self.bn5(self.cv5(img))))
        img = F.relu(self.bn6(self.cv6(img)))
        
        img = img.view(img.shape[0],-1)
        img = self.dropout(F.relu(self.fc1(img)))
        img = self.dropout(F.relu(self.fc2(img)))
        img = self.fc3(img)


        return img

