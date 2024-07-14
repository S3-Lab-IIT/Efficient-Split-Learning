from torchvision import models
import torch
import torch.nn as nn

"""
PARAMETER COUNTS
----------------
front: 11,160
center_front: 5,274,256
center_back: 14,561,832
back: 340,488
frozen: 5,285,416
trainable: 14,902,320
total: 20,187,736
"""

def efficientnetv2():
    m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    return m


class front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = efficientnetv2()
        self.f0 = m.features[0]
        self.f1 = m.features[1]
        
    def forward(self,x):
        x = self.f0(x)
        x = self.f1(x)
        return x
    

class center_front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = efficientnetv2()
        self.f2 = m.features[2]
        self.f3 = m.features[3]
        self.f4 = m.features[4]
        self.f5 = m.features[5]
        
    def forward(self,x):
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        return x
    

class center_back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = efficientnetv2()
        self.f6 = m.features[6]

    def freeze(self,epoch,pretrained):
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self,x):
        x = self.f6(x)
        return x
    

class back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = efficientnetv2()
        self.f7 = m.features[7]
        self.spatial_dropout = nn.Dropout2d(p=0.2)
        self.avgpool = m.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280,1)
        )
        
    def forward(self,x):
        x = self.f7(x)
        x = self.spatial_dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x