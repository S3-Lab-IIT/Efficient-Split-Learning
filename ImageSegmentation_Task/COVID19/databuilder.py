import pandas as pd
from pathlib import Path
import numpy as np
from config import covid19_path
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def normalize(img, maxval=255, reshape=False):
    """
    All credits to: https://github.com/mlmed/torchxrayvision
    Permalink: https://github.com/mlmed/torchxrayvision/blob/cd669b6af0279be8b2a6674b7366878a76f75fba/torchxrayvision/utils.py#L45
    Scales images to be roughly [-1024 1024].
    """

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


class COVID19Dataset:
    def __init__(self,cases,tfms):
        self.cases = cases
        self.tfms = tfms
        self.to_torch = A.Compose([ToTensorV2()])
    def __len__(self,):
        return len(self.cases)
    def __getitem__(self,idx):
        case = self.cases[idx]
        im, label = case['image'], case['label']
        im = np.array(Image.open(im).convert('RGB'))
        aug = self.tfms(image=im)
        im = aug['image']
        im = normalize(im)
        im = self.to_torch(image=im)['image']
        label = torch.tensor([label]).long()
        return {
            'image': im,
            'label': label
        }

class Covid19DataBuilder:
    def __init__(self,):
        self.thresholded_sites_path = Path('./Datasets/COVID19/splits_main.csv').resolve()
        self.thresholded_sites = pd.read_csv(self.thresholded_sites_path)
        self.full_data_size = len(self.thresholded_sites)
        self.data_dir = covid19_path
        self.img_size = 224
        

    def get_client_cases(self,client_id, pool=False):
        if pool:
            client = self.thresholded_sites
        else:
            client = self.thresholded_sites.query(f'client == {client_id}').reset_index(drop=True)
        train_cases = client.query(f"split == 'train'").reset_index(drop=True)
        valid_cases = client.query(f"split == 'test'").reset_index(drop=True)
        return train_cases, valid_cases
    
    def _make_dict(self,cases):
        client_dicts = []
        for i in range(len(cases)):
            client_dicts.append({
                'image': self.data_dir / f"{cases.loc[i,'filename']}",
                'label': int(cases.loc[i,'covid_19'])
            })
        return client_dicts
    
    def get_data_dict(self,client_id,pool):
        train_cases, valid_cases = self.get_client_cases(client_id,pool)

        train_dict = self._make_dict(train_cases)
        valid_dict = self._make_dict(valid_cases)

        return train_dict, valid_dict

    def get_data_transforms(self,):

        train_transforms = A.Compose([
            A.RandomScale(0.1),
            A.RandomBrightnessContrast(0.15, 0.1),
            A.HorizontalFlip(p=0.5),
            A.Resize(self.img_size, self.img_size),
        ])
        val_transforms = A.Compose([
            A.Resize(self.img_size,self.img_size)
        ])

        return train_transforms, val_transforms
    
    def get_datasets(self,client_id,cache=False,cache_rate=1.0,pool=False):
        train_files, valid_files = self.get_data_dict(client_id,pool)
        train_tfms, val_tfms = self.get_data_transforms()
    
        train_ds = COVID19Dataset(cases=train_files,tfms=train_tfms)
        val_ds = COVID19Dataset(cases=valid_files,tfms=val_tfms)

        return train_ds, val_ds


if __name__ == '__main__':
    ixi = Covid19DataBuilder()
    train_ds, val_ds = ixi.get_datasets(0)
    print(len(train_ds),len(val_ds))
    dl = torch.utils.data.DataLoader(train_ds,batch_size=16)
    print(len(dl))
    item = next(iter(dl))
    im,lb = item['image'], item['label']
    print(im.shape, lb.shape)