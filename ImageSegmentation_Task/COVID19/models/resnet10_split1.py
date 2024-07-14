import torch
import torch.nn as nn
from timm import create_model


"""
PARAMETER COUNTS
----------------
client front: 26,232
server center_front: 73,984
server center_back: 4,822,272
client back: 157,313
frozen: 100,216
trainable: 4,979,585
total: 5,079,801
"""

def get_model():
    m = create_model('resnet10t',pretrained=True,num_classes=1)
    return m


class front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = get_model()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.act1 = m.act1
        self.maxpool = m.maxpool

        for p in self.parameters():
            p.requires_grad = False

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        return x
    

class center_front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = get_model()
        self.l1 = m.layer1

        for p in self.parameters():
            p.requires_grad = False

    def forward(self,x):
        x = self.l1(x)
        return x
    

class center_back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = get_model()
        self.l2 = m.layer2
        self.l3 = m.layer3
        self.l4 = m.layer4

    def freeze(self,epoch,pretrained):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self,x):
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
    

class back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = get_model()
        self.conv_dw = nn.Conv2d(512,128,kernel_size=(7,7),stride=(1,1),padding=(3,3),groups=128)
        self.norm = nn.LayerNorm((128,))
        self.mlp = nn.Sequential(
            nn.Linear(128,512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512,128)
        )
        self.drop2d = nn.Dropout2d(0.3)
        self.pool = nn.AdaptiveAvgPool1d(128)
        self.head = nn.Linear(128,1)
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