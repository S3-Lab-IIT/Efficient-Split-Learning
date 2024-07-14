import torch
import torch.nn as nn
from timm import create_model

"""
MODEL PARAMETERS:
front 28768
center front 19035904
center back  trainable: 1052672, total: 4462592
back 16392
"""


def full_model():
    m = create_model('resnet50d',pretrained=True,num_classes=8)
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
        self.layer4_01 = m.layer4[:2]

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4_01(x)
        return x
    
class center_back(nn.Module):
    def __init__(self,):
        m = full_model()
        super().__init__()
        self.layer4_2 = m.layer4[-1]
        to_freeze = [
            self.layer4_2.conv1,
            self.layer4_2.bn1,
            self.layer4_2.act1, 
            self.layer4_2.conv2,
            self.layer4_2.bn2,
        ]
        for L in to_freeze:
            for p in L.parameters():
                p.requires_grad = False

    def freeze(self,epoch,pretrained):
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self,x):
        x = self.layer4_2(x)
        return x
    

class back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = full_model()
        self.gp = m.global_pool
        self.fc = m.fc
        
    def forward(self,x):
        x = self.gp(x)
        x = self.fc(x)
        return x