from torchvision import models
import torch.nn as nn

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
    m = models.resnet18(pretrained=True)
    return m


class front(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.act1 = m.relu
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

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
    

class center_back(nn.Module):
    def __init__(self,):
        super().__init__()
        m = resnet18()
        self.l3 = m.layer3
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
        self.conv_dw = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=512)
        self.conv_dw1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.bn = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fl = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(in_features=512, out_features=10, bias=True)
        
    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_dw1(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.fl(x)
        x = self.fc(x)
        return x
    
# Instantiate the model and print its architecture
front_model = front()
print("Front Model:")
print(front_model)

center_front_model = center_front()
print("\nCenter Front Model:")
print(center_front_model)

center_back_model = center_back()
print("\nCenter Back Model:")
print(center_back_model)

back_model = back()
print("\nBack Model:")
print(back_model)