from torch import nn
import torch

from Datasets.kits19.models.nnUNet.nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from Datasets.kits19.models.nnUNet.nnunet.network_architecture.initialization import InitWeights_He

"""
PARAMETER COUNT:
----------------
front: 194,976
center front: 8,298,624
center back: 21,792,768
back: 497,312
split full: 30,783,680
trainable: 22,290,080
frozen: 8,493,600
"""

class Baseline(Generic_UNet):
    def __init__(self):
        pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        conv_kernel_sizes = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
        super(Baseline, self).__init__(
            input_channels=1,
            base_num_features=32,
            num_classes=3,
            num_pool=5,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0, "inplace": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=True,
            convolutional_upsampling=True,
            max_num_features=None,
            basic_block=ConvDropoutNormNonlin,
        )

def get_full_model():
    model = Baseline()
    return model


# ------- SPLITS --------


class front(nn.Module):
    def __init__(self,input_channels=1,pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        full_model = get_full_model()
        model_state_dict = torch.load('/media/imroze/A8DCE5B8DCE580C4/Shreyas/medical_split_learning/Datasets/kits19/models/pretrained/best_new_model.pth')
        full_model.load_state_dict(model_state_dict)

        # conv_blocks_context, convolutional pooling in full model must be True
        self.front_contexts = full_model.conv_blocks_context[:2] # 1, 2
        
        for p in self.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        self.skips = [] # reset skips every forward pass
        for stacked_conv_layer in self.front_contexts:
            x = stacked_conv_layer(x)
            self.skips.append(x)
        return x
    

class center_front(nn.Module):
    def __init__(self,pretrained=False,skips=[]):
        super().__init__()
        full_model = get_full_model()
        self.pretrained = pretrained
        full_model = get_full_model()
        model_state_dict = torch.load('/media/imroze/A8DCE5B8DCE580C4/Shreyas/medical_split_learning/Datasets/kits19/models/pretrained/best_new_model.pth')
        full_model.load_state_dict(model_state_dict)

        # conv_blocks_context, convolutional pooling in full model must be True
        self.center_contexts = full_model.conv_blocks_context[2:-1] # 3,4,5
        """
        skips consist of both the front model skips and the center model skips
        after forward pass of front model, set center.skips = front.skips
        the same skips will be then shared with the back model for upsampling layers
        """
        self.skips = []

        for p in self.parameters():
            p.requires_grad = False

        
    def forward(self, x):
        for stacked_conv_layer in self.center_contexts:
            x = stacked_conv_layer(x)
            self.skips.append(x)
        
        return x
    

class center_back(nn.Module):
    def __init__(self,pretrained=False,skips=[]):
        super().__init__()
        full_model = get_full_model()
        self.pretrained = pretrained
        full_model = get_full_model()
        model_state_dict = torch.load('/media/imroze/A8DCE5B8DCE580C4/Shreyas/medical_split_learning/Datasets/kits19/models/pretrained/best_new_model.pth')
        full_model.load_state_dict(model_state_dict)

        self.skips = []
        
        self.final_context_layer = full_model.conv_blocks_context[-1] # 6

        # tu & conv_blocks_localization
        # tu: ConvTranspose3D ModuleList
        self.center_tu = full_model.tu[:3] # 1, 2, 3
        self.center_localizations = full_model.conv_blocks_localization[:3] # 1, 2, 3
        
        
    def forward(self, x):

        x = self.final_context_layer(x)

        # reverse skip connections.
        skips = self.skips[::-1]
        for u in range(len(self.center_tu)):
            x = self.center_tu[u](x)
            x = torch.cat((x, skips[u]), dim=1)
            x = self.center_localizations[u](x)
        
        return x
    
    def freeze(self, epoch, pretrained=False):
        for p in self.parameters():
            p.requires_grad = False




class back(nn.Module):
    def __init__(self,pretrained=False,skips=[]):
        super().__init__()
        self.pretrained=pretrained
        full_model = get_full_model()
        model_state_dict = torch.load('/media/imroze/A8DCE5B8DCE580C4/Shreyas/medical_split_learning/Datasets/kits19/models/pretrained/best_new_model.pth')
        full_model.load_state_dict(model_state_dict)

        # tu & conv_blocks_localization
        # tu: ConvTranspose3D ModuleList
        self.back_tu = full_model.tu[3:] # 4, 5
        self.back_localizations = full_model.conv_blocks_localization[3:] # 4, 5
        self.skips = skips
        
        # final 32 -> num_class layer
        self.final_seg_output = full_model.seg_outputs[-1]

        
    def forward(self, x):
        skips = self.skips[::-1][3:]
        for u in range(len(self.back_tu)):
            x = self.back_tu[u](x)
            x = torch.cat((x, skips[u]), dim=1)
            x = self.back_localizations[u](x)
            
        x = self.final_seg_output(x)
        return x
    




if __name__ == '__main__':
    f = front()
    cf = center_front()
    cb = center_back()
    b = back()
    
    x = torch.rand(2,1,96,96,96)
    
    fx = f(x)
    cf.skips = f.skips
    cfx = cf(fx)
    cb.skips = cf.skips
    cbx = cb(cfx)
    b.skips = cb.skips
    
    bx = b(cbx)


    print('skips shapes',[s.shape for s in b.skips])
    
    assert bx.shape == (2,3,96,96,96)
    
    f_params = sum([p.numel() for p in f.parameters() if p.requires_grad])
    cf_params = sum([p.numel() for p in cf.parameters() if p.requires_grad])
    cb_params = sum([p.numel() for p in cb.parameters() if p.requires_grad])
    b_params = sum([p.numel() for p in b.parameters() if p.requires_grad])
    
    full = (f_params + cf_params + cb_params + b_params)
    trainable = (cb_params + b_params)
    print(f'front: {f_params:,}\ncenter front: {cf_params:,}\ncenter back: {cb_params:,}\nback: {b_params:,}')
    print(f'split full: {full:,}')
    print(f'trainable: {trainable:,}')
    print(f'frozen: {(full-trainable):,}')

    print(bx.shape)
