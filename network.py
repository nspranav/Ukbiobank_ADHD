import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv3d(1, 16, 3, stride=(1,1,1), padding=0) #try increasing filter size
        self.bn1 = nn.BatchNorm3d(16)
        self.cv2 = nn.Conv3d(16, 32, 3, stride=(1,1,1), padding=0) #try increase filter size
        self.bn2 = nn.BatchNorm3d(32)
        self.cv3 = nn.Conv3d(32, 64, 3,stride=(1,1,1), padding=0)
        self.bn3 = nn.BatchNorm3d(64)
        self.cv4 = nn.Conv3d(64, 128, 3,stride=(1,1,1), padding=0)
        self.bn4 = nn.BatchNorm3d(128)
        self.cv5 = nn.Conv3d(128, 128, 3,stride=(1,1,1), padding=0)
        self.bn5 = nn.BatchNorm3d(128)
        self.cv6 = nn.Conv3d(128, 64, 1,stride=(1,1,1), padding=0)
        self.bn6 = nn.BatchNorm3d(64)
        self.cv7 = nn.Conv3d(64,64,1) 
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.2)
        self.dropout5 = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool3d(3)
        
    def forward(self, img):
        img = self.pool(F.relu(self.bn1(self.cv1(img))))
        img = self.pool(F.relu(self.bn2(self.cv2(img))))
        img = self.pool(F.relu(self.bn3(self.cv3(img))))
        img = self.pool(F.relu(self.bn4(self.cv4(img))))
        img = self.pool(F.relu(self.bn5(self.cv5(img))))
        #img = self.pool(F.relu(self.bn6(self.cv6(img))))
        #img = self.cv7(self.dropout(self.avgpool(img)))

        img = img.view(img.shape[0], -1)
        img = self.dropout(F.relu(self.fc1(img)))
        img = self.dropout(F.relu(self.fc2(img)))
        img = self.dropout(F.relu(self.fc3(img)))
        img = self.dropout(F.relu(self.fc4(img)))


        return img

