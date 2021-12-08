import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv3d(1, 32, 3, stride=1, padding=0) 
        self.bn1 = nn.BatchNorm3d(32)
        self.cv2 = nn.Conv3d(32, 64, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.cv3 = nn.Conv3d(64, 128, 3,stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(128)
        self.cv4 = nn.Conv3d(128, 256, 3,stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(256)
        self.cv5 = nn.Conv3d(256, 512, 3,stride=1, padding=0)
        self.bn5 = nn.BatchNorm3d(256)
        self.cv6 = nn.Conv3d(128, 128, 1,stride=1, padding=0)
        self.bn6 = nn.BatchNorm3d(128)
        self.cv7 = nn.Conv3d(64,64,1) 
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(1024, 1)
        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,16)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(16,1)
        self.dropout = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool3d(3)

        self.layer1 = nn.Sequential(self.cv1,self.pool,nn.ReLU(),self.bn1)
        self.layer2 = nn.Sequential(self.cv2,self.pool,nn.ReLU(),self.bn2)
        self.layer3 = nn.Sequential(self.cv3,self.pool,nn.ReLU(),self.bn3)
        self.layer4 = nn.Sequential(self.cv4,self.pool,nn.ReLU(),self.bn4)
        self.layer5 = nn.Sequential(self.cv5,self.pool,nn.ReLU(),self.bn5)

    def forward(self, img):

        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        img = self.layer5(img)
        #img = F.relu(self.pool(self.cv6(img)))
        #img = F.relu(self.cv7(self.dropout(self.avgpool(img))))

        img = img.view(img.shape[0], -1)
        img = self.dropout(self.fc1(img))
        #img = self.dropout(self.fc2(img))
        #img = self.dropout(F.relu(self.fc3(img)))
        #img = self.dropout(F.relu(self.fc4(img)))
        ##img = self.dropout(F.relu(self.fc5(img)))
        #img = self.dropout(F.relu(self.fc6(img)))

        return img

