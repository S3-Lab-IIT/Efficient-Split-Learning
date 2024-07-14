import torch
import torch.nn as nn
import numpy as np
from config import isic19_path
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from loss_fn import WeightedFocalLoss
from timm import create_model
from sklearn.metrics import balanced_accuracy_score

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
        label = torch.tensor([label]).float()
        return {
            'image': im,
            'label': label
        }
    
train_transforms = A.Compose([
            A.RandomScale(0.07),
            A.Rotate(50),
            A.RandomBrightnessContrast(0.15, 0.1),
            A.Flip(p=0.5),
            A.Affine(shear=0.1),
            A.Resize(224, 224),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])
val_transforms = A.Compose([
    A.Resize(224,224),
    A.CenterCrop(224, 224),
    A.Normalize(always_apply=True),
    ToTensorV2()
])

p = Path('./Datasets/ISIC2019/splits.csv').resolve()
df = pd.read_csv(p)
main_cases = df.query(f"fold == 'train'").reset_index(drop=True)
test_cases = df.query(f"fold == 'test'").reset_index(drop=True)
train_cases, valid_cases = train_test_split(main_cases,test_size=0.1,shuffle=True)
train_cases.reset_index(drop=True,inplace=True)
valid_cases.reset_index(drop=True,inplace=True)

def make_dict(cases):
    client_dicts = []
    for i in range(len(cases)):
        client_dicts.append({
            'image': isic19_path / (cases.loc[i,'image']+'.jpg'),
            'label': int(cases.loc[i,'target'])
        })
    return client_dicts

train_cases = ISICDataset(make_dict(train_cases),tfms=train_transforms)
valid_cases = ISICDataset(make_dict(valid_cases),tfms=val_transforms)
test_cases = ISICDataset(make_dict(test_cases),tfms=val_transforms)

train_dl = torch.utils.data.DataLoader(train_cases,batch_size=64,shuffle=True,num_workers=2)
valid_dl = torch.utils.data.DataLoader(valid_cases,batch_size=64,shuffle=False,num_workers=2)
test_dl = torch.utils.data.DataLoader(test_cases,batch_size=64,shuffle=False,num_workers=2)

device = 'cuda'

loss_fn = WeightedFocalLoss()

class Resnet50(nn.Module):
    def __init__(self,):
        super().__init__()
        m = create_model('resnet50d',pretrained=True,num_classes=8)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.act1 = m.act1
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        for l in [self.conv1,self.bn1,self.act1,self.maxpool,self.layer1,self.layer2,self.layer3]:
            for p in l.parameters():
                p.requires_grad=False

        self.conv_dw = nn.Conv2d(2048,256,kernel_size=(7,7),stride=(1,1),padding=(3,3),groups=256)
        self.norm = nn.LayerNorm((256,))
        self.mlp = nn.Sequential(
            nn.Linear(256,1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024,256)
        )
        self.drop2d = nn.Dropout2d(0.3)
        self.pool = nn.AdaptiveAvgPool1d(256)
        self.head = nn.Linear(256,8)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = self.drop2d(x)
        x = torch.flatten(x,1)
        x = self.pool(x)
        x = self.head(x)
        return x
        

model = Resnet50().to(device=device)
optim = torch.optim.AdamW(model.parameters(),lr=1e-3)

def metric(preds,gt):
    gt = gt.reshape(-1).numpy()
    preds = np.argmax(preds.numpy(), axis=1)
    return balanced_accuracy_score(gt, preds)


n = 30


tl = []
mt = []

vl = []
mv = []

best_mv = 0.
mtest = []

for ep in tqdm(range(n)):

    train_gt = []
    train_pd = []
    
    valid_gt = []
    valid_pd = []

    model.train()
    
    running_tl = 0.
    for batch in tqdm(train_dl):
        img, lb = batch['image'].to(device), batch['label'].to(device)

        train_gt.append(lb.detach().cpu())
        preds = model(img)
        train_pd.append(preds.detach().cpu())

        loss =loss_fn(preds,lb)
        loss.backward()
        optim.step()
        optim.zero_grad()
        running_tl += loss.item()
        
    tl.append(sum(running_tl)/len(train_dl))
    m = metric(
        preds=torch.vstack(train_pd).numpy(),
        gt=torch.vstack(train_gt).numpy()
    )
    mt.append(m)

    with torch.no_grad():

        model.eval()

        running_vl = 0.
        for batch in tqdm(valid_dl):
            img, lb = batch['image'].to(device), batch['label'].to(device)

            valid_gt.append(lb.detach().cpu())
            preds = model(img)
            valid_pd.append(preds.detach().cpu())

            loss =loss_fn(preds,lb)
            tl.append(loss.item())
            running_vl += loss.item()
    
        vl.append(sum(running_vl)/len(valid_dl))

        m = metric(
            preds=torch.vstack(valid_pd).numpy(),
            gt=torch.vstack(valid_gt).numpy()
        )
        mv.append(m)

        print('\n\n','='*10,'\n\n')
        print(f"EPOCH {ep+1}")
        print(f'train loss: {tl} | valid loss: {vl}')
        print(f'train metric: {mt} | valid metric: {mv}')

        if m > best_mv:

            # inference
            test_gt = []
            test_pd = []

            for batch in tqdm(test_dl):
                img, lb = batch['image'].to(device), batch['label'].to(device)

                test_gt.append(lb.detach().cpu())
                preds = model(img)
                test_pd.append(preds.detach().cpu())

            m = metric(
                preds=torch.vstack(test_pd).numpy(),
                gt=torch.vstack(test_gt).numpy()
            )

            mtest.append(m)


            print('TEST METRIC BEST:',m)

        print('\n\n','='*10,'\n\n')