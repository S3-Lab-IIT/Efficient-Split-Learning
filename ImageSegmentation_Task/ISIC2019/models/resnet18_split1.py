import torch
import torch.nn as nn
from timm import create_model

"""
PARAMETER COUNTS
----------------
client front: 28,768
server center_front: 1,592,576
server center_back: 9,574,400
client back: 293,640
frozen: 1,621,344
trainable: 9,868,040
total: 11,489,384
"""


def resnet18():
    m = create_model('resnet18d',pretrained=True,num_classes=8)
    return m


class front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.act1 = m.act1
        self.pool = m.maxpool

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        return x
    

class center_front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.l1 = m.layer1
        self.l2 = m.layer2
        self.l3_b0 = m.layer3[0]

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3_b0(x)
        return x
    

class center_back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.l3 = m.layer3[1:]
        self.l4 = m.layer4

    def freeze(self,epoch,pretrained):
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        x = self.l3(x)
        x = self.l4(x)
        return x
    

class back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.conv_dw = nn.Conv2d(512,512,kernel_size=(7,7),stride=(1,1),padding=(3,3),groups=512)
        self.norm = nn.LayerNorm((512,))
        self.mlp = nn.Sequential(
            nn.Linear(512,256),
            nn.GELU(),
            nn.Linear(256,512)
        )
        self.pool = m.global_pool
        self.fc = m.fc
        
    def forward(self, x):
        skip = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = x+skip
        x = self.pool(x)
        x = self.fc(x)
        return x