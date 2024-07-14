"""
Medical Split Learning on Patched Camelyon16 dataset based on Flamby splits
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# PFSL
from utils.random_clients_generator import generate_random_clients
from utils.connections import send_object
from utils.argparser import parse_arguments
from utils.merge import merge_weights
from ImageSegmentation_Task.PCam.databuilder import PCamDataBuilder
from ImageSegmentation_Task.PCam.pcam_client import Client
from ImageSegmentation_Task.PCam.pcam_server import ConnectedClient

from config import WANDB_KEY

# flags
os.environ['CUDA_LAUNCH_BLOCKING']='1'



"""
PCam CLASS
"""

class PCamTrainer:

    def seed(self):
        """seed everything along with cuDNN"""
        seed = self.args.seed
        random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


    def init_clients_with_data(self,):
        """
        initialize PFSL clients: (id, class: Client)
        along with their individual data based on Flamby splits
        """
        assert self.args.number_of_clients <= 2, 'max clients for pcam is 2'
        self.num_clients = self.args.number_of_clients if not self.pooling_mode else 1

        self.clients = generate_random_clients(self.num_clients,Client)

        if self.pooling_mode:
            key = list(self.clients.keys())[0]
            self.clients['pooled_client'] = self.clients.pop(key) 

        self.client_ids = list(self.clients.keys())


        for idx, (c_id, client) in enumerate(self.clients.items()):

            train_ds, test_ds = self.pcam.get_datasets(client_id=idx, pool=self.pooling_mode)

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
        splits are made from Resnet50/18, effnetv2_small
        and optimizer for each split: Adam / AdamW / Novograd
        """

        model = importlib.import_module(self.import_module)
        pretrained = self.args.pretrained
        lr = self.args.client_lr

        for c_id, client in self.clients.items():
            client.device = self.device
            
            client.front_model = model.front().to(self.device)
            client.front_model.eval()
            
            client.back_model = model.back().to(self.device)
            client.back_optimizer = AdamW(client.back_model.parameters(), lr=lr,weight_decay=3e-2)
            client.back_scheduler = StepLR(client.back_optimizer,step_size=4,gamma=0.1)

        print(f'initialized client-side model splits front&back and their optimizers')


    def init_clients_server_copy(self,):
        """
        for each client, there is a server copy of the center model
        initialized using class: ConnectedClient(client_id, connection)

        the center model and its optimizer is initialized as well.
        """

        model = importlib.import_module(self.import_module)
        pretrained = self.args.pretrained
        lr = self.args.server_lr

        self.sc_clients = dict()
        for c_id in self.client_ids:
            self.sc_clients[c_id] = ConnectedClient(id=c_id,conn=None)

        for c_id, sc_client in self.sc_clients.items():
            sc_client.device = self.device
            
            sc_client.center_front_model = model.center_front().to(self.device)
            sc_client.center_front_model.eval()

            sc_client.center_back_model = model.center_back().to(self.device)
            sc_client.center_optimizer = AdamW(sc_client.center_back_model.parameters(), lr=lr,weight_decay=3e-2)
            sc_client.center_scheduler = StepLR(sc_client.center_optimizer,step_size=4,gamma=0.)

        print(f'initialized server-side model splits center_front, center_back and center_back optimizer')


    def _create_save_dir(self,):
        self.save_dir = Path(f'./saved_models/{self.args.dataset}/key_value_mode/model_split{self.args.split}')
        self.save_dir.mkdir(exist_ok=True,parents=True)

    
    def remove_frozen_models(self,):
        """
        - the client-side forward model and the server-side center_front model are unused after
        the key-value store mappings are generated.
        - the models are rather saved to disk and moved to CPU during runtime to save GPU
        """
        for c_id in self.client_ids:
            # client-side front model
            front_state_dict = self.clients[c_id].front_model.state_dict()
            torch.save(front_state_dict, self.save_dir / f'client_{c_id}_front.pth')
            self.clients[c_id].front_model.cpu()
            # server-side center_front model
            center_front_state_dict = self.sc_clients[c_id].center_front_model.state_dict()
            torch.save(center_front_state_dict, self.save_dir / f'client_{c_id}_center_front.pth')
            self.sc_clients[c_id].center_front_model.cpu()



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
            sample_lens.append(len(self.clients[c_id].train_dataset) * self.args.kv_factor)
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
        - append 0 for train_f1/test_f1 per client list
        - assign iterators per client from train/test dataloader
        """
        num_iters = {c_id:0 for c_id in self.client_ids}
        for c_id, client in self.clients.items():
    
            if dl=='train':
                client.train_f1.append(0)
                client.iterator = iter(client.train_DataLoader)
                client.num_iterations = len(client.train_DataLoader)
                len_keys = len(self.sc_clients[c_id].activation_mappings)
                num_iters[c_id] = int(ceil(len_keys / client.train_batch_size))
            elif dl=='test':
                client.test_f1.append(0)
                client.test_iterator = iter(client.test_DataLoader)
                client.num_test_iterations = len(client.test_DataLoader)
                num_iters[c_id] = len(client.test_DataLoader)

        return num_iters


    def store_forward_mappings_kv(self,mode='train'):
        """
        kv: key-value store {data_key:np.Array}\n
        since client-side front model & server-side center-front model is "frozen"
        - we only need the outputs from both the models once
        - the outputs are stored in activation mappings, each output with its own index
        - the targets are stored in target mappings, each target with its own index
        
        these values are reused every epoch / refreshed once in a while, this is done for both train and test mappings
        """

        if mode=='train':

            # create iterators for initial forward pass of training phase
            for c_id, client in self.clients.items():
                client.num_iterations = len(client.train_DataLoader)
                
            # forward client-side front model which sets activation and target mappings.
            for c_id, client in tqdm(self.clients.items()):
                for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="client front"):
                    if client.data_key % len(client.train_dataset) == 0:
                        client.iterator = iter(client.train_DataLoader)
                    client.forward_front_key_value()

            # send activation mappings to server-side for center models use.
            for c_id, client in self.clients.items():
                self.sc_clients[c_id].activation_mappings = client.activation_mappings
                self.sc_clients[c_id].all_keys = list(client.activation_mappings.keys())
                
            # select random activation mappings of length=batch_size
            # forward server-side center_front model
            for c_id, sc_client in tqdm(self.sc_clients.items()):    
                num_iters = int(ceil(len(sc_client.all_keys) / self.clients[c_id].train_batch_size))
                for it in tqdm(range(num_iters),desc="server center_front"):
                    # key selection
                    sc_client.current_keys=list(np.random.choice(sc_client.all_keys, min(self.clients[c_id].train_batch_size, len(sc_client.all_keys)), replace=False))
                    sc_client.update_all_keys()
                    # choosing activations from client-side and moving them to server-side
                    self.clients[c_id].activations1=torch.Tensor(np.array([sc_client.activation_mappings[x] for x in sc_client.current_keys])).to(self.device)
                    self.clients[c_id].remote_activations1=self.clients[c_id].activations1.detach().requires_grad_(True)
                    sc_client.remote_activations1=self.clients[c_id].remote_activations1
                    # forward center_front
                    sc_client.forward_center_front()

        else:

            # create iterators for initial forward pass of testing phase
            for c_id, client in self.clients.items():
                client.test_iterator=iter(client.test_DataLoader)
                client.num_test_iterations = len(client.test_DataLoader)

            # [TEST] forward client-side front model which sets activation and target mappings.
            for c_id, client in tqdm(self.clients.items()):
                for it in tqdm(range(client.num_test_iterations)):
                    client.forward_front_key_value_test()

            # [TEST] send activation, mappings to server-side for center models use.
            for c_id, client in self.clients.items():
                self.sc_clients[c_id].test_activation_mappings = client.test_activation_mappings
                self.sc_clients[c_id].all_keys = list(client.test_activation_mappings.keys())

            # [TEST] select random activation mappings of length=test_batch_size
            # [TEST] forward server-side center_front model
            for c_id, sc_client in tqdm(self.sc_clients.items()):
                for it in tqdm(range(self.clients[c_id].num_test_iterations)):
                    # key selection
                    sc_client.current_keys=list(np.random.choice(sc_client.all_keys, min(self.clients[c_id].test_batch_size, len(sc_client.all_keys)), replace=False))
                    sc_client.update_all_keys()
                    # choosing activations from client-side and moving them to server-side
                    self.clients[c_id].activations1=torch.Tensor(np.array([sc_client.test_activation_mappings[x] for x in sc_client.current_keys])).to(self.device)
                    self.clients[c_id].remote_activations1=self.clients[c_id].activations1.detach().requires_grad_(True)
                    sc_client.remote_activations1=self.clients[c_id].remote_activations1
                    # forward center_front
                    sc_client.forward_center_front_test()


    def populate_key_value_store(self,):
        """
        - resets key-value store for client and server
        - populates key-value store kv_factor no. of times
        """

        for c_id in self.client_ids:
            self.clients[c_id].data_key = 0
            self.clients[c_id].test_data_key = 0

        print('generating training samples in key-value store...')
        self.store_forward_mappings_kv(mode='train')
        print('generating testing samples in key-value store...')
        self.store_forward_mappings_kv(mode='test')


    def train_one_epoch(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - step back optimizer
                - merge grads
                - step center_back optimizer
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
            - merge model weights across clients (center_back & back)
        """

        print(f"training {epoch}...\n\n")
        num_iters = self.create_iters(dl='train')
        max_iters = max(num_iters.values())
        self.overall_f1['train'].append(0)

        # set keys
        for c_id, sc_client in self.sc_clients.items():
            sc_client.all_keys = list(sc_client.activation_mappings.keys())

        # per iteration, run the following:
        for it in tqdm(range(max_iters)):

            # forward server-side center_back model with activations
            for c_id, sc_client in self.sc_clients.items():
                if num_iters[c_id] != 0:
                    sc_client.current_keys=list(np.random.choice(sc_client.all_keys, min(self.clients[c_id].train_batch_size, len(sc_client.all_keys)), replace=False))
                    sc_client.update_all_keys()
                    sc_client.middle_activations=torch.Tensor(np.array([sc_client.activation_mappings[x] for x in sc_client.current_keys])).to(self.device).detach().requires_grad_(True)

                    sc_client.forward_center_back()

            # forward client-side back model with activations
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    client.current_keys = self.sc_clients[c_id].current_keys
                    client.remote_activations2 = self.sc_clients[c_id].remote_activations2

                    client.forward_back()
                    client.set_targets()

            # calculate train loss
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    client.calculate_loss(mode='train')
                    wandb.log({'train step loss': client.loss.item()})

            # backprop (back model) in client equivalent for client.backward_back()
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    client.loss.backward()

            # backprop (center model) in sc_client
            for c_id, sc_client in self.sc_clients.items():
                if num_iters[c_id] != 0:
                    sc_client.activations2 = self.clients[c_id].remote_activations2
                    sc_client.backward_center()

            # step optim and zero grad client back model
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    client.step_back()
                    client.zero_grad_back()

            # step optim and zero grad sc_client center model
            for c_id, sc_client in self.sc_clients.items():
                if num_iters[c_id] != 0:
                    sc_client.center_optimizer.step()
                    sc_client.center_optimizer.zero_grad()

            # train f1 of every client in the current epoch in the current batch
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    f1=client.calculate_train_metric()
                    client.train_f1[-1] += f1 
                    print("train f1 per iteration: ", f1)
                    wandb.log({f'train f1 / iter: client {c_id}':f1.item()})

            # reduce num_iters per client by 1
            # training loop will only execute for a client if iters are left 
            for c_id in self.client_ids:
                if num_iters[c_id] != 0:
                    num_iters[c_id] -= 1

        for c_id in self.client_ids:
            self.clients[c_id].back_scheduler.step()
            self.sc_clients[c_id].center_scheduler.step()

        # calculate epoch metrics
        AUROCs = []
        avg_loss = 0
        for c_id, client in self.clients.items():
            num_iters = len(client.activation_mappings.keys())
            client.train_f1[-1] /= int(ceil(num_iters / client.train_batch_size))
            client.train_loss /= int(ceil(num_iters / client.train_batch_size))
            avg_loss += client.train_loss
            self.overall_f1['train'][-1] += client.train_f1[-1]
            AUROC_client = client.get_main_metric(mode='train') 
            AUROCs.append(AUROC_client)
            wandb.log({f'avg train f1 {c_id}': client.train_f1[-1].item()})
            wandb.log({f'auroc train {c_id}':AUROC_client})
            wandb.log({f'avg train loss {c_id}': client.train_loss})
            client.train_loss = 0 # reset for next epoch


        # calculate epoch metrics across clients
        AUROC = np.array(AUROCs).mean()
        self.overall_f1['train'][-1] /= self.num_clients
        print("train f1: ", self.overall_f1['train'][-1])
        wandb.log({'avg train f1 all clients': self.overall_f1['train'][-1].item()})
        wandb.log({'avg train auroc all clients': AUROC})
        wandb.log({'avg train loss all clients': avg_loss / self.num_clients})

        if not self.pooling_mode:
            # merge model weights (center and back)
            self.merge_model_weights(epoch)

            
    @torch.no_grad()
    def test_one_epoch(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
        """

        num_iters = self.create_iters(dl='test')
        max_iters = max(num_iters.values())
        self.overall_f1['test'].append(0)

        for c_id, client in self.clients.items():
            client.pred = []
            client.y = []

        # set keys
        for c_id, sc_client in self.sc_clients.items():
            sc_client.all_keys = list(sc_client.test_activation_mappings.keys())

        # per iteration in testing epoch, do the following:
        for it in tqdm(range(max_iters)):

            # forward server-side center_back model with activations
            for c_id, sc_client in self.sc_clients.items():
                if num_iters[c_id] != 0:
                    sc_client.current_keys=list(np.random.choice(sc_client.all_keys, min(self.clients[c_id].test_batch_size, len(sc_client.all_keys)), replace=False))
                    sc_client.update_all_keys()
                    sc_client.middle_activations=torch.Tensor(np.array([sc_client.test_activation_mappings[x] for x in sc_client.current_keys])).to(self.device).detach().requires_grad_(True)

                    sc_client.forward_center_back()

            # forward client-side back model with activations
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    client.current_keys = self.sc_clients[c_id].current_keys
                    client.remote_activations2 = self.sc_clients[c_id].remote_activations2

                    client.forward_back()
                    client.set_test_targets()

            # calculate test loss
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    client.calculate_loss(mode='test')
                    wandb.log({'test step loss': client.loss.item()})

            # test f1 of every client in the current epoch in the current batch
            for c_id, client in self.clients.items():
                if num_iters[c_id] != 0:
                    f1=client.calculate_test_metric()
                    client.test_f1[-1] += f1 
                    print("test f1 per iteration: ", f1)
                    wandb.log({f'test f1 / iter: client {c_id}':f1.item()})

            # reduce num_iters per client by 1
            # testing loop will only execute for a client if iters are left 
            for c_id in self.client_ids:
                if num_iters[c_id] != 0:
                    num_iters[c_id] -= 1

        
        # calculate epoch metrics
        avg_loss = 0
        AUROCs = []
        for c_id, client in self.clients.items():
            client.test_f1[-1] /= len(client.test_DataLoader)
            client.test_loss /= len(client.test_DataLoader)

            # calculate remaining metrics
            avg_loss += client.test_loss
            AUROC_client = client.get_main_metric(mode='test')
            AUROCs.append(AUROC_client)
            self.overall_f1['test'][-1] += client.test_f1[-1]
            wandb.log({f'avg test f1 {c_id}': client.test_f1[-1].item()})
            wandb.log({f'auroc test {c_id}':AUROC_client})
            wandb.log({f'avg test loss {c_id}': client.test_loss})
            client.test_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        AUROC = np.array(AUROCs).mean()
        self.overall_f1['test'][-1] /= self.num_clients
        print("test f1: ", self.overall_f1['test'][-1])
        wandb.log({'avg test f1 all clients': self.overall_f1['test'][-1].item()})
        wandb.log({'avg test auroc all clients': AUROC})
        wandb.log({'avg test loss all clients': avg_loss / self.num_clients})

        # max f1 score achieved on test dataset
        if(self.overall_f1['test'][-1]> self.max_f1['f1']):
            self.max_f1['f1']=self.overall_f1['test'][-1]
            self.max_f1['epoch']=epoch
            print(f"MAX test f1 score: {self.max_f1['f1']} @ epoch {self.max_f1['epoch']}")
            wandb.log({
                'max test f1 score':self.max_f1['f1'].item(),
                'max_test_f1_epoch':self.max_f1['epoch']
            })
            # save at best model
            return True
        
        return False # don't save


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


    def clear_cache(self,):
        gc.collect()
        torch.cuda.empty_cache()


    def fit(self,):
        """
        - trains and tests the models for given num. of epochs
        """

        if self.pooling_mode:
            print('\n\nPOOLING MODE: ENABLED!')

        self._create_save_dir()

        # disabled freeing GPU mem since key-value store needs to be refreshed
        # print('freeing some GPU...')
        # self.remove_frozen_models()

        # if key value store refresh rate = 0, it is disabled
        if self.args.kv_refresh_rate == 0:
            self.populate_key_value_store()
            self.clear_cache()
        
        
        print(f'{"-"*25}\n\ncommence training...\n\n')

        for epoch in tqdm(range(self.args.epochs)):

            # if key value store refresh rate != 0, it is enabled
            if self.args.kv_refresh_rate != 0:
                if epoch % self.kv_refresh_rate == 0:
                    print(f'\npreparing key value store for the next {self.kv_refresh_rate} epochs\n\n')
                    self.populate_key_value_store()
                    self.clear_cache()

            if self.args.personalize:
                if epoch == self.args.p_epoch:
                    self.personalization_mode = True
                    self.personalize(epoch)

            wandb.log({'epoch':epoch})

            for c_id in self.client_ids:
                self.clients[c_id].back_model.train()
                self.sc_clients[c_id].center_back_model.train()

            self.train_one_epoch(epoch)
            self.clear_cache()

            for c_id in self.client_ids:
                self.clients[c_id].back_model.eval()
                self.sc_clients[c_id].center_back_model.eval()

            is_best = self.test_one_epoch(epoch)

            if is_best:
                self.save_models()

            self.clear_cache()

        # final metrics
        print(f'\n\n\n{"::"*10}BEST METRICS{"::"*10}')
        print("Training Mean f1 Score: ", self.overall_f1['train'][self.max_f1['epoch']])
        print("Maximum Test Mean f1 Score: ", self.max_f1['f1'])

        self.run.finish()



    def __init__(self,args):
        """
        implementation of PFSL training & testing simulation on Patched Camelyon16 dataset
            - Patched Camelyon16: Histopathology Image Classification
            - model: ResNet

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

        self.import_module = f"Datasets.PCam.models.{self.args.model}_split{self.args.split}"

        self.pooling_mode = self.args.pool

        # refresh key-value store every N epochs
        self.kv_refresh_rate = self.args.kv_refresh_rate

        wandb.login(key=WANDB_KEY)
        self.run = wandb.init(
            project='medical_split_learning_key_value_pcam',
            config=vars(self.args),
            job_type='train',
            mode='online' if self.log_wandb else 'disabled'
        )

        self.seed()

        self.pcam = PCamDataBuilder()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.overall_f1 = {
            'train': [],
            'test': []
        }

        self.max_f1 = {
            'f1': 0,
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
    trainer = PCamTrainer(args)
    trainer.fit()