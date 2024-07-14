import numpy as np
import pandas as pd
from pathlib import Path
from config import pcam_path
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py

class PCamDataset:
    def __init__(self,cases,tfms,h5):
        self.cases = cases
        self.tfms = tfms
        self.x = h5
    def __len__(self,):
        return len(self.cases)
    def __getitem__(self,idx):
        case = self.cases[idx]
        im, label = case['image'], case['label']
        im = self.x[idx]
        aug = self.tfms(image=im)
        im = aug['image']
        label = torch.tensor([label]).long()
        return {
            'image': im,
            'label': label
        }

        
class PCamDataBuilder:
    def __init__(self,):
        self.train_csv_path = Path('./Datasets/PCam/splits_train_tiny.csv').resolve()
        self.valid_csv_path = Path('./Datasets/PCam/splits_test_tiny.csv').resolve()
        self.train_csv = pd.read_csv(self.train_csv_path)
        self.valid_csv = pd.read_csv(self.valid_csv_path)
        self.full_data_size = len(self.train_csv) + len(self.valid_csv)
        self.data_dir = pcam_path
        self.img_size = 224
        self.train_ims = h5py.File(str(self.data_dir/'training_split.h5'),'r')['x']
        self.valid_ims = h5py.File(str(self.data_dir/'validation_split.h5'),'r')['x']
        

    def get_client_cases(self,client_id, pool=False):
        tc = self.train_csv
        vc = self.valid_csv
        if not pool:
            train_cases = tc.query(f'hospital == {client_id}').reset_index(drop=True)
            valid_cases = vc.query(f'hospital == {client_id}').reset_index(drop=True)
        else:
            train_cases = tc
            valid_cases = vc            

        return train_cases, valid_cases
    
    def _make_dict(self,cases):
        client_dicts = []
        for i in range(len(cases)):
            client_dicts.append({
                'image': int(cases.loc[i,'idx']),
                'label': int(cases.loc[i,'tumor_patch'])
            })
        return client_dicts

    
    def get_data_dict(self,client_id,pool):
        train_cases, valid_cases = self.get_client_cases(client_id,pool)

        train_dict = self._make_dict(train_cases)
        valid_dict = self._make_dict(valid_cases)

        return train_dict, valid_dict

    def get_data_transforms(
            self,
    ):

        train_transforms = A.Compose([
            A.Resize(self.img_size, self.img_size, always_apply=True),
            A.RandomRotate90(),
            A.Flip(p=0.5),
            A.Normalize(always_apply=True),
            A.CenterCrop(self.img_size, self.img_size, always_apply=True),
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(self.img_size,self.img_size,always_apply=True),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])

        return train_transforms, val_transforms
    
    def get_datasets(self,client_id,cache=False,cache_rate=1.0,pool=False):
        train_files, valid_files = self.get_data_dict(client_id,pool)
        train_tfms, val_tfms = self.get_data_transforms()
        
        train_ds = PCamDataset(cases=train_files,tfms=train_tfms,h5=self.train_ims)
        val_ds = PCamDataset(cases=valid_files,tfms=val_tfms,h5=self.valid_ims)

        return train_ds, val_ds


if __name__ == '__main__':
    pcam = PCamDataBuilder()
    train_ds, val_ds = pcam.get_datasets(0)
    print(len(train_ds),len(val_ds))
    dl = torch.utils.data.DataLoader(train_ds,batch_size=128)
    print(len(dl))
    item = next(iter(dl))
    im,lb = item['image'], item['label']
    print(im.shape, lb.shape)
    print(lb)