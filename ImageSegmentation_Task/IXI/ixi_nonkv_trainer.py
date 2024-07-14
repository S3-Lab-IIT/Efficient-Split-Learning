"""
Medical Split Learning on IXI-2019 dataset based on Flamby splits
"""

"""
IMPORTS
"""

# inbuilt
import os
import sys
import random
from math import ceil
import string
import requests, threading, time, socket, datetime
import multiprocessing
import copy
import importlib
import gc
from pathlib import Path

# usual
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm.auto import tqdm
import wandb


# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD

# PFSL
from utils.random_clients_generator import generate_random_clients
from utils.connections import send_object
from utils.argparser import parse_arguments
from utils.merge import merge_weights
from ImageSegmentation_Task.IXI.databuilder import IXIDataBuilder
from ImageSegmentation_Task.IXI.ixi_client import Client
from ImageSegmentation_Task.IXI.ixi_server import ConnectedClient

from config import WANDB_KEY

# flags
os.environ['CUDA_LAUNCH_BLOCKING']='1'



"""
KITS CLASS
"""

class IXITrainer:

    def seed(self):
        """seed everything along with cuDNN"""
        seed = self.args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def init_clients_with_data(self,):
        """
        initialize PFSL clients: (id, class: Client)
        along with their individual data based on Flamby splits
        """
        assert self.args.number_of_clients <= 3, 'max clients for IXI is 3'
        self.num_clients = self.args.number_of_clients if not self.pooling_mode else 1

        self.clients = generate_random_clients(self.num_clients,Client)

        if self.pooling_mode:
            key = list(self.clients.keys())[0]
            self.clients['pooled_client'] = self.clients.pop(key) 

        self.client_ids = list(self.clients.keys())


        for idx, (c_id, client) in enumerate(self.clients.items()):

            train_ds, test_ds = self.ixi.get_datasets(client_id=idx, pool=self.pooling_mode)

            client.train_dataset = train_ds
            client.test_dataset = test_ds

            print(f"client {c_id} -> #train {len(train_ds)} #test: {len(test_ds)}")

            client.create_DataLoader(
                self.train_batch_size,
                self.test_batch_size
            )

        print(f'generated {self.num_clients} clients with data')


    def init_client_models_optims(self, input_channels=1):
        """
        initialize client-side model splits
        splits: front, back, center
        splits are made from MONAI 3DUNet model (./models/3dunet_split*.py)
        and optimizer for each split: Adam / AdamW / Novograd
        """

        model = importlib.import_module(self.import_module)
        pretrained = self.args.pretrained
        lr = self.args.client_lr

        for c_id, client in self.clients.items():
            
            client.front_model = model.front(input_channels, pretrained=pretrained).to(self.device)
            client.front_model.eval()
            
            client.back_model = model.back(pretrained=pretrained).to(self.device)
            client.back_optimizer = AdamW(client.back_model.parameters(), lr=lr)  

        print(f'initialized client-side model splits front&back and their optimizers')


    def init_clients_server_copy(self,):
        """
        for each client, there is a server copy of the center model
        initialized using class: ConnectedClient(client_id, connection)

        the center front and center back models and center back optimizer is initialized as well.
        """

        model = importlib.import_module(self.import_module)
        pretrained = self.args.pretrained
        lr = self.args.server_lr

        self.sc_clients = dict()
        for c_id in self.client_ids:
            self.sc_clients[c_id] = ConnectedClient(id=c_id,conn=None)

        for c_id, sc_client in self.sc_clients.items():
            sc_client.device = self.device
            
            sc_client.center_front_model = model.center_front(pretrained=pretrained).to(self.device)
            sc_client.center_front_model.eval()

            sc_client.center_back_model = model.center_back(pretrained=pretrained).to(self.device)
            sc_client.center_optimizer = AdamW(sc_client.center_back_model.parameters(), lr=lr)

        print(f'initialized server-side model splits center_back and center_back optimizer')


    def personalize(self,epoch):
        """
        personalization: 
            freeze all layers of center model in server copy clients
        """
        print(f'personalizing, freezing server copy center model @ epoch {epoch}')
        for c_id, sc_client in self.sc_clients.items():
            sc_client.center_back_model.freeze(epoch,pretrained=self.args.pretrained)


    def merge_model_weights(self,epoch):
        """
        - merge weights and distribute over all server-side center_back models
        - In the personalisation phase merging of weights of the back model layers is stopped:
            - merge weights and distribute over all client-side back models
        """
        params = []
        sample_lens = []
        for c_id, sc_client in self.sc_clients.items():
            params.append(copy.deepcopy(sc_client.center_back_model.state_dict()))
            sample_lens.append(len(self.clients[c_id].train_dataset))
        # pfsl merge weights util
        w_glob = merge_weights(params, sample_lens)

        for c_id, sc_client in self.sc_clients.items():
            sc_client.center_back_model.load_state_dict(w_glob)

        if not self.personalization_mode:

            params = []
            sample_lens = []
            for c_id, client in self.clients.items():
                params.append(copy.deepcopy(client.back_model.state_dict()))
                sample_lens.append(len(client.train_dataset))
            # pfsl merge weights util
            w_glob_cb = merge_weights(params,sample_lens)
    
            for c_id, client in self.clients.items():
                client.back_model.load_state_dict(w_glob_cb)

        del params, sample_lens
        
        
    def create_iters(self,dl='train'):
        """
        -> append 0 for train_dice/test_dice per client list
        -> assign self.iterator per client from train/test dataloader
        -> return num iters per client
        """
        num_iters = {c_id:0 for c_id in self.client_ids}
        for c_id, client in self.clients.items():
    
            if dl=='train':
                client.train_dice.append(0)
                client.iterator = iter(client.train_DataLoader)
                num_iters[c_id] = len(client.train_DataLoader)
            elif dl=='test':
                client.test_dice.append(0)
                client.iterator = iter(client.test_DataLoader)
                num_iters[c_id] = len(client.test_DataLoader)

        return num_iters


    def train_one_epoch(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward front model of client
                - forward center model of server
                - forward back model of client
                - step back model
                - merge grads
                - step center model
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
            - merge model weights across clients (center & back)
        """
        
        print(f"training {epoch}...\n\n")

        num_train_iters = self.create_iters(dl='train')
        max_iters = max(num_train_iters.values())

        self.overall_dice['train'].append(0)

        for it in tqdm(range(max_iters)):

            # set server copy clients skips=[]
            for c_id, sc_client in self.sc_clients.items():
                if num_train_iters[c_id] != 0:
                    sc_client.skips=[]

            # forward front model of all clients
            for c_id, client in self.clients.items():
                if num_train_iters[c_id] != 0:
                    client.forward_front()

            # set sc_client.remote_activations1 to client.remote_activations1
            # set sc_client skips to [client.remote_activations1]
            # forward sc_client center model with the updated skip connections
            for c_id, sc_client in self.sc_clients.items():
                if num_train_iters[c_id] != 0:
                    sc_client.remote_activations1 = self.clients[c_id].remote_activations1
                    sc_client.skips=[]
                    # sc_client.skips.append(self.clients[c_id].remote_activations1)
                    sc_client.skips=self.clients[c_id].skips
                    sc_client.center_front_model.skips = sc_client.skips
                    sc_client.forward_center()

            # set client.remote_activations2 to sc_client.remote_activations2
            # forward client back model
            for c_id, client in self.clients.items():
                if num_train_iters[c_id] != 0:
                    client.back_model.skips = self.sc_clients[c_id].skips
                    client.remote_activations2 = self.sc_clients[c_id].remote_activations2
                    client.forward_back()

            # calculate train loss
            for c_id, client in self.clients.items():
                if num_train_iters[c_id] != 0:
                    client.calculate_loss(mode='train')
                    if self.log_wandb:  wandb.log({'train step loss': client.loss.item()})

            # backprop (back model) in client equivalent for client.backward_back()
            for c_id, client in self.clients.items():
                if num_train_iters[c_id] != 0:
                    client.loss.backward()

            # backprop (center model) in sc_client
            for c_id, sc_client in self.sc_clients.items():
                if num_train_iters[c_id] != 0:
                    sc_client.activations2 = self.clients[c_id].remote_activations2
                    sc_client.backward_center()

            # step optim and zero grad client back model
            for c_id, client in self.clients.items():
                if num_train_iters[c_id] != 0:
                    client.step_back()
                    client.zero_grad_back()


            # step optim and zero grad sc_client center model
            for c_id, sc_client in self.sc_clients.items():
                if num_train_iters[c_id] != 0:
                    sc_client.center_optimizer.step()
                    sc_client.center_optimizer.zero_grad()

            # train dice of every client in the current epoch in the current batch
            for c_id, client in self.clients.items():
                if num_train_iters[c_id] != 0:
                    dice=client.calculate_train_dice_kits()
                    client.train_dice[-1] += dice 
                    print("train dice per iteration: ", dice)
                    if self.log_wandb:  wandb.log({f'train dice / iter: client {c_id}':dice.item()})

            # reduce num_train_iters per client by 1
            # training loop will only execute for a client if iters are left 
            for c_id in self.client_ids:
                if num_train_iters[c_id] != 0:
                    num_train_iters[c_id] -= 1

            
        avg_loss = 0
        # calculate epoch metrics
        for c_id, client in self.clients.items():
            client.train_dice[-1] /= len(client.train_DataLoader)
            client.train_loss /= len(client.train_DataLoader)
            avg_loss += client.train_loss
            self.overall_dice['train'][-1] += client.train_dice[-1]
            if self.log_wandb:  wandb.log({f'avg train dice {c_id}': client.train_dice[-1].item()})
            if self.log_wandb:  wandb.log({f'avg train loss {c_id}': client.train_loss})
            client.train_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        self.overall_dice['train'][-1] /= self.num_clients
        print("train dice: ", self.overall_dice['train'][-1])
        if self.log_wandb:  wandb.log({'avg train dice all clients': self.overall_dice['train'][-1].item()})
        if self.log_wandb:  wandb.log({'avg train loss all clients': avg_loss / self.num_clients})

        if not self.pooling_mode:
            # merge model weights (center and back)
            self.merge_model_weights(epoch)


    @torch.no_grad()
    def test_one_epoch(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward front model of client
                - forward center model of server
                - forward back model of client
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
            - keep track of maximum test dice achieved while training 
        """
        
        num_test_iters = self.create_iters(dl='test')
        max_iters = max(num_test_iters.values())

        self.overall_dice['test'].append(0)

        for c_id, client in self.clients.items():
            client.pred = []
            client.y = []

        for it in tqdm(range(max_iters)):

            # set server copy clients skips=[]
            for c_id, sc_client in self.sc_clients.items():
                if num_test_iters[c_id] != 0:
                    sc_client.skips=[]

            # forward front model of all clients
            for c_id, client in self.clients.items():
                if num_test_iters[c_id] != 0:
                    client.forward_front()

            # set sc_client.remote_activations1 to client.remote_activations1
            # set sc_client skips to [client.remote_activations1]
            # forward sc_client center model with the updated skip connections
            for c_id, sc_client in self.sc_clients.items():
                if num_test_iters[c_id] != 0:
                    sc_client.remote_activations1 = self.clients[c_id].remote_activations1
                    sc_client.skips=[]
                    # sc_client.skips.append(self.clients[c_id].remote_activations1)
                    sc_client.skips=self.clients[c_id].skips
                    sc_client.center_front_model.skips = sc_client.skips
                    sc_client.forward_center()

            # set client.remote_activations2 to sc_client.remote_activations2
            # forward client back model
            for c_id, client in self.clients.items():
                if num_test_iters[c_id] != 0:
                    client.back_model.skips = self.sc_clients[c_id].skips
                    client.remote_activations2 = self.sc_clients[c_id].remote_activations2
                    client.forward_back()

            # calculate test loss
            for c_id, client in self.clients.items():
                if num_test_iters[c_id] != 0:
                    client.calculate_loss(mode='test')
                    if self.log_wandb:  wandb.log({'test step loss': client.loss.item()})

            # test dice of every client in the current epoch in the current batch
            for c_id, client in self.clients.items():
                if num_test_iters[c_id] != 0:
                    dice=client.calculate_test_dice_kits()
                    client.test_dice[-1] += dice 
                    print("test dice per iteration: ", dice)
                    if self.log_wandb:  wandb.log({f'test dice / iter: client {c_id}':dice.item()})

            # reduce num_test_iters per client by 1
            # testing loop will only execute for a client if iters are left 
            for c_id in self.client_ids:
                if num_test_iters[c_id] != 0:
                    num_test_iters[c_id] -= 1


        avg_loss = 0
        # calculate epoch metrics
        for c_id, client in self.clients.items():
            client.test_dice[-1] /= len(client.test_DataLoader)
            client.test_loss /= len(client.test_DataLoader)
            avg_loss += client.test_loss
            self.overall_dice['test'][-1] += client.test_dice[-1]
            if self.log_wandb:  wandb.log({f'avg test dice {c_id}': client.test_dice[-1].item()})
            if self.log_wandb:  wandb.log({f'avg test loss {c_id}': client.test_loss})
            client.test_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        self.overall_dice['test'][-1] /= self.num_clients
        print("test dice: ", self.overall_dice['test'][-1])
        if self.log_wandb:  wandb.log({'avg test dice all clients': self.overall_dice['test'][-1].item()})
        if self.log_wandb:  wandb.log({'avg test loss all clients': avg_loss / self.num_clients})

        # max dice score achieved on test dataset
        if(self.overall_dice['test'][-1]> self.max_dice['dice']):
            self.max_dice['dice']=self.overall_dice['test'][-1]
            self.max_dice['epoch']=epoch
            print(f"MAX test dice score: {self.max_dice['dice']} @ epoch {self.max_dice['epoch']}")
            if self.log_wandb:  wandb.log({
                'max test dice score':self.max_dice['dice'].item(),
                'max_test_dice_epoch':self.max_dice['epoch']
            })


    def clear_cache(self,):
        gc.collect()
        torch.cuda.empty_cache()

    def _create_save_dir(self,):
        self.save_dir = Path(f'./saved_models/{self.args.dataset}/normal_mode/model_split{self.args.split}')
        self.save_dir.mkdir(exist_ok=True,parents=True)

    def save_models(self,):
        """
        save client-side back and server-side center_back models to disk
        """
        for c_id in self.client_ids:
            # client-side front model
            front_state_dict = self.clients[c_id].front_model.state_dict()
            torch.save(front_state_dict, self.save_dir / f'client_{c_id}_front.pth')
            # server-side center_front model
            center_front_state_dict = self.sc_clients[c_id].center_front_model.state_dict()
            torch.save(center_front_state_dict, self.save_dir / f'client_{c_id}_center_front.pth')
            # server-side center_back model
            center_back_state_dict = self.sc_clients[c_id].center_front_model.state_dict()
            torch.save(center_back_state_dict, self.save_dir / f'client_{c_id}_center_back.pth')
            # client-side back model
            back_state_dict = self.clients[c_id].front_model.state_dict()
            torch.save(back_state_dict, self.save_dir / f'client_{c_id}_back.pth')


    def fit(self,):
        """
        - trains and tests the models for given num. of epochs
        """

        if self.pooling_mode:
            print('POOLING MODE: ENABLED!')

        self._create_save_dir()

        for epoch in tqdm(range(self.args.epochs)):

            if self.args.personalize:
                if epoch == self.args.p_epoch:
                    self.personalization_mode = True
                    self.personalize(epoch)

            if self.log_wandb:  wandb.log({'epoch':epoch})

            for c_id in self.client_ids:
                self.clients[c_id].back_model.train()
                self.sc_clients[c_id].center_back_model.train()

            self.train_one_epoch(epoch)
            self.clear_cache()

            for c_id in self.client_ids:
                self.clients[c_id].back_model.eval()
                self.sc_clients[c_id].center_back_model.eval()

            self.test_one_epoch(epoch)
            self.clear_cache()

        # final metrics
        print(f'\n\n\n{"::"*40}')
        print("Training Mean Dice Score: ", self.overall_dice['train'][self.max_dice['epoch']])
        print("Maximum Test Mean Dice Score: ", self.max_dice['dice'])

        if self.log_wandb:
            self.run.finish()



    def __init__(self,args):
        """
        implementation of PFSL training & testing simulation on IXI-Tiny dataset
            - IXI-Tiny: 3D Brain segmentation
            - model: MONAI 3DUNet

        initialize everything:
            - data
            - wandb logging
            - metrics
            - device
            - batch sizes
            - flags
            - models
            - pooling mode: train all samples together
        """

        self.args = args
        self.log_wandb = self.args.wandb

        self.import_module = f"Datasets.IXI.models.{self.args.model}_split{self.args.split}"

        if self.log_wandb:
            if self.log_wandb:  wandb.login(key=WANDB_KEY)
            self.run = wandb.init(
                project='medical_split_learning_normal',
                config=vars(self.args),
                job_type='train'
            )

        self.seed()

        self.ixi = IXIDataBuilder()

        self.pooling_mode = self.args.pool

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.overall_dice = {
            'train': [],
            'test': []
        }

        self.max_dice = {
            'dice': 0,
            'epoch': -1
        }

        self.train_batch_size = self.args.batch_size
        self.test_batch_size = self.args.test_batch_size

        self.personalization_mode = False

        self.init_clients_with_data()

        self.init_client_models_optims()

        self.init_clients_server_copy()



if __name__ == '__main__':
    args = parse_arguments()
    trainer = IXITrainer(args)
    trainer.fit()