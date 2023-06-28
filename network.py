import torch.nn as nn
import torch.nn.functional as F
import torch

class Network(nn.Module):
    """
    aaasdasda 
    """
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv3d(1, 32, 3, stride=1, padding=0) #121*145*121
        self.bn1 = nn.BatchNorm3d(32)
        self.cv2 = nn.Conv3d(32, 64, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.cv3 = nn.Conv3d(64, 128, 3,stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(128)
        self.cv4 = nn.Conv3d(128, 256, 3,stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(256)
        self.cv5 = nn.Conv3d(256, 256, 3,stride=1, padding=0) # 256*1*2
        self.bn5 = nn.BatchNorm3d(256) 
        
        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(512, 1)
        #self.fc2 = nn.Linear(256,6)

        self.d3d = nn.Dropout3d(0.2)
        

        self.layer2 = nn.Sequential(self.cv2,self.bn2,self.pool,nn.ReLU())
        self.layer3 = nn.Sequential(self.cv3,self.bn3,self.pool,nn.ReLU())
        self.layer4 = nn.Sequential(self.cv4,self.bn4,self.pool,nn.ReLU())
        self.layer1 = nn.Sequential(self.cv1,self.bn1,self.pool,nn.ReLU())
        self.layer5 = nn.Sequential(self.cv5,self.bn5,self.pool,nn.ReLU())

        self.convs = nn.Sequential(self.layer1,self.layer2,self.layer3,
                        self.layer4, self.layer5)
        self.convs.apply(Network.init_weights)
        #self.classifier = nn.Sequential(nn.Dropout(),self.fc1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, img,data=None):

        img = self.convs(img)

        #img = F.relu(self.pool(self.cv6(img)))
        #img = F.relu(self.cv7(self.dropout(self.avgpool(img))))

        img = img.view(img.shape[0], -1)

        # Adding sex data to the convoluted data
        if data is not None:
            with torch.no_grad():
                img = torch.cat((img,torch.unsqueeze(data,1)),dim=1)
        
        img = self.fc1(self.dropout(img))
        #img = self.dropout(F.relu(self.fc3(img)))
        #img = self.dropout(F.relu(self.fc4(img)))
        #img = self.dropout(F.relu(self.fc5(img)))
        #img = self.dropout(F.relu(self.fc6(img)))

        return img

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m,nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            #m.bias.data.fill_(0.01)

