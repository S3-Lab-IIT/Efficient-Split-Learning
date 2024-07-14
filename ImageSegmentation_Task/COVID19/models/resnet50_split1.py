import torch
import torch.nn as nn
from timm import create_model

"""
PARAMETER COUNTS
----------------
client front: 28,768
server center_front: 8,533,760
server center_back: 14,964,736
client back: 4,96,520
frozen: 8,562,528
trainable: 15,461,256
total: 24,023,784
"""

def full_model():
    m = create_model('resnet50d',pretrained=True,num_classes=1)
    return m

class front(nn.Module):
    def __init__(self,):
        m = full_model()
        super().__init__()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.act1 = m.act1
        self.maxpool = m.maxpool
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        return x
    

class center_front(nn.Module):
    def __init__(self,):
        m = full_model()
        super().__init__()
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    

class center_back(nn.Module):
    def __init__(self,):
        m = full_model()
        super().__init__()
        self.layer4 = m.layer4

    def freeze(self,epoch,pretrained):
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self,x):
        x = self.layer4(x)
        return x
    

class back(nn.Module):
    def __init__(self,):
        m = full_model()
        super().__init__()
        self.conv_dw = nn.Conv2d(2048,256,kernel_size=(7,7),stride=(1,1),padding=(3,3),groups=256)
        self.norm = nn.LayerNorm((256,))
        self.mlp = nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024,256)
        )
        self.drop2d = nn.Dropout2d(0.3)
        self.pool = nn.AdaptiveAvgPool1d(256)
        self.head = nn.Linear(256,1)
        
    def forward(self,x):
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = self.drop2d(x)
        x = torch.flatten(x,1)
        x = self.pool(x)
        x = self.head(x)
        
        return x