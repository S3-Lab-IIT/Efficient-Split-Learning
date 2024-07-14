import pandas as pd
from pathlib import Path
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    ScaleIntensityd,
    RandAffined,
    RandSpatialCropd,
    CenterSpatialCropd
)
from monai.data.dataset import Dataset, CacheDataset
import numpy as np
from config import ixitiny_path
import torch
from sklearn.model_selection import train_test_split

class IXIDataBuilder:
    def __init__(self,):
        self.thresholded_sites_path = Path('./Datasets/IXI/splits.csv').resolve()
        self.thresholded_sites = pd.read_csv(self.thresholded_sites_path)
        self.full_data_size = len(self.thresholded_sites)
        self.data_dir = ixitiny_path
        

    def get_client_cases(self,client_id, pool=False):
        if pool:
            client = self.thresholded_sites
        else:
            client = self.thresholded_sites.query(f'Manufacturer == {client_id}').reset_index(drop=True)
        main_cases = client.query(f"Split == 'train'")['filename'].to_list()
        test_cases = client.query(f"Split == 'test'")['filename'].to_list()
        return main_cases, test_cases
    
    def _make_dict(self,cases):
        return [
            {
                'image': self.data_dir/f'image/{c}_image.nii.gz',
                'label': self.data_dir/f'label/{c}_label.nii.gz'
            } for c in cases
        ]
    
    def get_data_dict(self,client_id,pool):
        main_cases, test_cases = self.get_client_cases(client_id,pool)

        main_dict = self._make_dict(main_cases)
        test_dict = self._make_dict(test_cases)

        train_dict, valid_dict = train_test_split(main_dict,test_size=0.1,shuffle=True)

        return train_dict, valid_dict, test_dict

    def get_data_transforms(
            self,
            voxel_spacing=(1.5,1.5,2.0),
            spatial_size=(96,96,96)
    ):

        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], 
                     pixdim=voxel_spacing, 
                     mode=("bilinear", "nearest")
                    ),
            # RandAffined(
            #     keys=["image", "label"],
            #     mode=("bilinear", "nearest"),
            #     prob=1.0,
            #     spatial_size=spatial_size,
            #     translate_range=(25,25,2),
            #     rotate_range=(np.pi / 32, np.pi / 32, np.pi / 16),
            #     scale_range=(0.15, 0.15, 0.15),
            #     padding_mode="border",
            # ),
            # RandSpatialCropd(
            #     keys=["image","label"],
            #     roi_size=spatial_size
            # ),
            ResizeWithPadOrCropd(keys=["image", "label"],
            spatial_size=spatial_size,
            mode='constant')
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], 
                     pixdim=voxel_spacing, 
                     mode=("bilinear", "nearest")
                    ),
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=spatial_size
            ),
            ResizeWithPadOrCropd(keys=["image", "label"],
            spatial_size=spatial_size,
            mode='constant')
        ])

        return train_transforms, val_transforms
    
    def get_datasets(self,client_id,cache=False,cache_rate=1.0,pool=False):
        train_files, valid_files, test_files = self.get_data_dict(client_id,pool)
        train_tfms, val_tfms = self.get_data_transforms()
        
        train_ds = Dataset(data=train_files, transform=train_tfms)
        val_ds = Dataset(data=valid_files, transform=val_tfms)
        test_ds = Dataset(data=test_files, transform=val_tfms)

        return train_ds, val_ds, test_ds


if __name__ == '__main__':
    ixi = IXIDataBuilder()
    train_ds, val_ds = ixi.get_datasets(0)
    print(len(train_ds),len(val_ds))
    dl = torch.utils.data.DataLoader(train_ds,batch_size=16)
    print(len(dl))
    item = next(iter(dl))
    im,lb = item['image'], item['label']
    print(im.shape, lb.shape)