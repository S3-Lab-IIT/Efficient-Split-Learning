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
    Rand3DElasticd,
    RandSpatialCropd,
    CenterSpatialCropd
)
import monai.transforms as M
from monai.data.dataset import Dataset, CacheDataset
import numpy as np
from config import kits19_path
from sklearn.model_selection import train_test_split

class KITSDataBuilder:
    def __init__(self,):
        self.thresholded_sites_path = Path('./Datasets/kits19/splits.csv').resolve()
        self.thresholded_sites = pd.read_csv(self.thresholded_sites_path)
        self.full_data_size = len(self.thresholded_sites)
        self.data_dir = kits19_path
        

    def get_client_cases(self,client_id, pool=False):
        if pool:
            client = self.thresholded_sites
        else:
            client = self.thresholded_sites.query(f'site_ids == {client_id}').reset_index(drop=True)
        main_cases = client.query(f"train_test_split == 'train'")['case_ids'].to_list()
        test_cases = client.query(f"train_test_split == 'test'")['case_ids'].to_list()
        return main_cases, test_cases

    
    def _make_dict(self,cases):
        return [
            {
                'image': self.data_dir/c/'imaging.nii.gz',
                'label': self.data_dir/c/'segmentation.nii.gz'
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
            voxel_spacing=(2.90,1.45,1.45),
            spatial_size=(96,96,96),
            is_dynamic=False
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
            RandSpatialCropd(
                keys=["image","label"],
                roi_size=spatial_size
            ),
            ResizeWithPadOrCropd(keys=["image", "label"],
            spatial_size=spatial_size,
            mode='constant'),
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=spatial_size,
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)
            #     ),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=1.0,
                spatial_size=spatial_size,
                translate_range=(40, 40, 2),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            ),
            # Rand3DElasticd(
            #     keys=["image", "label"],
            #     mode=("bilinear", "nearest"),
            #     prob=1.0,
            #     sigma_range=(5, 8),
            #     magnitude_range=(100, 200),
            #     spatial_size=spatial_size,
            #     translate_range=(50, 50, 2),
            #     rotate_range=(np.pi / 36, np.pi / 36, np.pi),
            #     scale_range=(0.15, 0.15, 0.15),
            # )
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

        if is_dynamic:
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
                        ResizeWithPadOrCropd(keys=["image", "label"],
                        spatial_size=spatial_size,
                        mode='constant'),
                    ])


        return train_transforms, val_transforms
    
    def get_dynamic_transforms(self,):
        return M.Compose([
            M.EnsureChannelFirst(channel_dim=0),
            M.RandAffine(
                translate_range=(40, 40, 20),
                rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                scale_range=(0.3, 0.3, 0.3),
                padding_mode="border",
            )
        ])
    
    def get_datasets(self,client_id,cache=False,cache_rate=1.0,pool=False,is_dynamic=False):
        train_files, valid_files, test_files = self.get_data_dict(client_id,pool)
        train_tfms, val_tfms = self.get_data_transforms(is_dynamic=is_dynamic)
        
        train_ds = Dataset(data=train_files, transform=train_tfms)
        val_ds = Dataset(data=valid_files, transform=val_tfms)
        test_ds = Dataset(data=test_files, transform=val_tfms)

        return train_ds, val_ds, test_ds






if __name__ == '__main__':
    
    from utils.random_clients_generator import generate_random_clients
    from Datasets.kits19.kits_client import Client

    print('Generating random clients...', end='')
    clients = generate_random_clients(6, Client=Client)
    client_ids = list(clients.keys())    
    print('Done')

    print(f'Random client ids:{str(client_ids)}')

    print('Initializing clients...')
    client_idx=0
    print("clients: ", clients)

    kits = KITSDataBuilder()
    train_batch_size = 2 # fixed for now
    test_batch_size = 2 # fixed for now

    for _, client in clients.items():

        train_ds,val_ds,_ = kits.get_datasets(client_id=client_idx)
        client.train_dataset = train_ds
        client.test_dataset = val_ds

        print(len(train_ds),len(val_ds))

        client.create_DataLoader(train_batch_size, test_batch_size)

       
        client_idx+=1
        print("client_idx: ", client_idx)
    
    print('Client Intialization complete.')    
    # Train and test data intialisation complete
