import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.networks.blocks.convolutions import Convolution, ResidualUnit

"""
PARAMETER COUNTS
----------------
front: 63412
center_front: 1107268
center_back: 3463300
back: 175934
frozen: 1170680
trainable: 3639234
full: 4809914
"""

def get_full_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=[16,32,64,128,256],
        strides=[2,2,2,2],
        num_res_units=2,
        norm=Norm.BATCH
    )
    return model


class front(nn.Module):
    def __init__(self,input_channels=1,pretrained=True,skips=[]):
        super().__init__()

        full_model = get_full_model()
        model_state_dict = torch.load('/home/shreyas/SPLIT_LEARNING/medical_split_learning/Datasets/IXI/models/pretrained/model.pt',map_location=torch.device('cpu'))
        full_model.load_state_dict(model_state_dict)
        full_model = full_model.model

        self.res1 = full_model[0]
        self.res2 = full_model[1].submodule[0]
        
        self.skips = skips

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        self.skips = [] # reset skips every forward pass
        x1 = self.res1(x)
        x2 = self.res2(x1)
        self.skips.extend([x1,x2])
        return x2
    

class center_front(nn.Module):
    def __init__(self,pretrained=True,skips=[]):
        super().__init__()

        full_model = get_full_model()
        model_state_dict = torch.load('/home/shreyas/SPLIT_LEARNING/medical_split_learning/Datasets/IXI/models/pretrained/model.pt',map_location=torch.device('cpu'))
        full_model.load_state_dict(model_state_dict)
        full_model = full_model.model

        self.res3 = full_model[1].submodule[1].submodule[0]
        self.res4 = full_model[1].submodule[1].submodule[1].submodule[0]
        
        self.skips = skips

        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        x3 = self.res3(x)
        x4 = self.res4(x3)
        self.skips.extend([x3,x4])
        return x4
    

class center_back(nn.Module):
    def __init__(self,pretrained=True,skips=[]):
        super().__init__()
        
        full_model = get_full_model()
        model_state_dict = torch.load('/home/shreyas/SPLIT_LEARNING/medical_split_learning/Datasets/IXI/models/pretrained/model.pt',map_location=torch.device('cpu'))
        full_model.load_state_dict(model_state_dict)
        full_model = full_model.model

        self.res5 = full_model[1].submodule[1].submodule[1].submodule[1].submodule
        self.sc_seq = full_model[1].submodule[1].submodule[1].submodule[2]
        self.skips = skips
        
    def forward(self, x):
        x5 = self.res5(x)
        
        skips = self.skips[::-1]
        
        r5_cat = torch.cat([x5, skips[0]],dim=1)
        x = self.sc_seq(r5_cat)
        
        return x
    

class back(nn.Module):
    def __init__(self,pretrained=True,skips=[]):
        super().__init__()
        
        full_model = get_full_model()
        model_state_dict = torch.load('/home/shreyas/SPLIT_LEARNING/medical_split_learning/Datasets/IXI/models/pretrained/model.pt',map_location=torch.device('cpu'))
        full_model.load_state_dict(model_state_dict)
        full_model = full_model.model

        self.sc_seq4 = full_model[1].submodule[1].submodule[2]
        self.sc_seq3 = full_model[1].submodule[2]
        self.sc_seq2 = self.final_layer()
        self.skips = skips

    def final_layer(self,):
        conv = Convolution(
            spatial_dims=3,
            in_channels=32,
            out_channels=3,
            strides=(2,2,2),
            kernel_size=(3,3,3),
            padding=(1,1,1),
            output_padding=(1,1,1),
            act="PRELU",
            dropout=0.,
            norm="INSTANCE",
            is_transposed=True
        )
        ru = ResidualUnit(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding=(1,1,1),
            last_conv_only=True,
            subunits=1
        )
        return nn.Sequential(conv,ru)
        
    def forward(self, x):
        
        skips = self.skips[::-1]
        
        r4_cat = torch.cat([x, skips[1]],dim=1)
        x = self.sc_seq4(r4_cat)
        
        r3_cat = torch.cat([x, skips[2]],dim=1)
        x = self.sc_seq3(r3_cat)
        
        r2_cat = torch.cat([x, skips[3]],dim=1)
        x = self.sc_seq2(r2_cat)
        
        
        return x
    

if __name__ == '__main__':

    p1 = front()
    p2 = center_front()
    p3 = center_back()
    p4 = back()

    params = lambda m: sum([p.numel() for p in m.parameters()])

    x = torch.rand(2,1,96,96,96)
    
    x1 = p1(x)
    
    p2.skips = p1.skips
    x2 = p2(x1)
    
    p3.skips = p2.skips
    x3 = p3(x2)
    
    p4.skips = p3.skips
    x4 = p4(x3)
    
    print(x4.shape)

    assert x4.shape == (2,3,96,96,96)    

    print('skips')
    for s in p2.skips:
        print(f'\t{s.shape}')
        
    full = params(get_full_model())
    par1 = params(p1)
    par2 = params(p2)
    par3 = params(p3)
    par4 = params(p4)
    s = par1+par2+par3+par4
    print('client:front',par1)
    print('server:center_front',par2)
    print('server:center_back',par3)
    print('client:back',par4)
    print('frozen',par1+par2)
    print('trainable',par3+par4)
    print('full',s)