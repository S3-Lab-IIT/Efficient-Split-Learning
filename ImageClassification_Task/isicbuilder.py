import numpy as np
import pandas as pd
from pathlib import Path
from config import isic19_path
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from sklearn.model_selection import train_test_split

class ISICDataset:
    def __init__(self,cases,tfms):
        self.cases = cases
        self.tfms = tfms
    def __len__(self,):
        return len(self.cases)
    def __getitem__(self,idx):
        case = self.cases[idx]
        im, label = case['image'], case['label']
        im = np.array(Image.open(im))
        aug = self.tfms(image=im)
        im = aug['image']
        label = torch.tensor([label]).long()
        return {
            'image': im,
            'label': label
        }

        
class ISICDataBuilder:
    def __init__(self,):
        self.thresholded_sites_path = Path('./Datasets/ISIC2019/splits.csv').resolve()
        self.thresholded_sites = pd.read_csv(self.thresholded_sites_path)
        self.full_data_size = len(self.thresholded_sites)
        self.data_dir = isic19_path
        self.img_size = 224
        

    def get_client_cases(self,client_id, pool=False):
        if pool:
            client = self.thresholded_sites
        else:
            client = self.thresholded_sites.query(f'center == {client_id}').reset_index(drop=True)
        main_cases = client.query(f"fold == 'train'").reset_index(drop=True)
        test_cases = client.query(f"fold == 'test'").reset_index(drop=True)
        return main_cases, test_cases
    
    def _make_dict(self,cases):
        client_dicts = []
        for i in range(len(cases)):
            client_dicts.append({
                'image': self.data_dir / (cases.loc[i,'image']+'.jpg'),
                'label': int(cases.loc[i,'target'])
            })
        return client_dicts

    
    def get_data_dict(self,client_id,pool):
        main_cases, test_cases = self.get_client_cases(client_id,pool)

        main_dict = self._make_dict(main_cases)
        test_dict = self._make_dict(test_cases)
        
        main_targets = [d['label'] for d in main_dict]
        train_dict, valid_dict = train_test_split(main_dict,test_size=0.125,shuffle=True,stratify=main_targets)


        return train_dict, valid_dict, test_dict

    def get_data_transforms(
            self,
    ):

        train_transforms = A.Compose([
            A.RandomScale(0.07),
            A.Rotate(50),
            A.RandomBrightnessContrast(0.15, 0.1),
            A.Flip(p=0.5),
            A.Affine(shear=0.1),
            A.Resize(self.img_size, self.img_size),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])
        val_transforms = A.Compose([
            A.Resize(self.img_size,self.img_size),
            A.CenterCrop(self.img_size, self.img_size),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])

        return train_transforms, val_transforms
    
    def get_datasets(self,client_id,cache=False,cache_rate=1.0,pool=False):
        train_files, valid_files, test_files = self.get_data_dict(client_id,pool)
        train_tfms, val_tfms = self.get_data_transforms()
        
        train_ds = ISICDataset(cases=train_files,tfms=train_tfms)
        val_ds = ISICDataset(cases=valid_files,tfms=val_tfms)
        test_ds = ISICDataset(cases=test_files,tfms=val_tfms)

        return train_ds, val_ds, test_ds


if __name__ == '__main__':
    isic = ISICDataBuilder()
    train_ds, val_ds, test_ds = isic.get_datasets(0)
    print(len(train_ds),len(val_ds), len(test_ds))
    dl = torch.utils.data.DataLoader(train_ds,batch_size=16)
    print(len(dl))
    item = next(iter(dl))
    im,lb = item['image'], item['label']
    print(im.shape, lb.shape)