"""
Efficient Split Learning on CIFAR/FMNIST/DR/ISIC-2019 dataset.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmetrics.functional import f1_score


# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# discriminator model
from .models.discriminator import Discriminator


#plot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

# PFSL
from utils.random_clients_generator import generate_random_clients
from utils.connections import send_object
from utils.argparser import parse_arguments
from utils.merge import merge_weights
from ImageClassification_Task.cifarbuilder import CIFAR10DataBuilder
from ImageClassification_Task.ic_client import Client
from ImageClassification_Task.ic_server import ConnectedClient

from config import WANDB_KEY

# flags
os.environ['CUDA_LAUNCH_BLOCKING']='1'



"""
ImageClassification IC CLASS
"""

class ICTrainer:

    def seed(self):
        """seed everything along with cuDNN"""
        seed = self.args.seed
        random.seed(seed)

    def init_clients_with_data(self,):
        """
        initialize PFSL clients: (id, class: Client)
        along with their individual data based on Flamby splits
        """
        assert self.args.number_of_clients <= 10, 'max clients for isic is 6'
        self.num_clients = self.args.number_of_clients if not self.pooling_mode else 1
        self.clients = generate_random_clients(self.num_clients,Client)
        if self.pooling_mode:
            key = list(self.clients.keys())[0]
            self.clients['pooled_client'] = self.clients.pop(key) 
        self.client_ids = list(self.clients.keys())
        
        self.clients_threshold = dict()
        for c_id, _ in self.clients.items():
            self.clients_threshold[c_id]=0
        
        data = []
        for idx, (c_id, client) in enumerate(self.clients.items()):
            train_ds, test_ds, main_test_ds = self.cifar_builder.get_datasets(client_id=idx, pool=self.pooling_mode)
            client.train_dataset = train_ds
            client.test_dataset = test_ds
            client.main_test_dataset = main_test_ds
            print(f"client {c_id} -> #train {len(train_ds)} #valid: {len(test_ds)} #test: {len(main_test_ds)}")
            client.create_DataLoader(
                self.train_batch_size,
                self.test_batch_size
            )
            try:
                train_class_counts = Counter(item['label'].item() for item in train_ds)
                test_class_counts = Counter(item['label'].item() for item in test_ds)
                main_test_class_counts = Counter(item['label'].item() for item in main_test_ds)
            except KeyError as e:
                print(f"KeyError: {e} not found in dataset items. Inspecting item structure for alternative keys or attributes...")
                raise
            
            # Calculate class proportions
            def calculate_proportions(class_counts):
                total = sum(class_counts.values())
                return {k: v / total for k, v in class_counts.items()}

            train_class_proportions = calculate_proportions(train_class_counts)
            test_class_proportions = calculate_proportions(test_class_counts)
            main_test_class_proportions = calculate_proportions(main_test_class_counts)
            
            print(f"Client {c_id} class distribution (proportions):")
            print(f"  Train: {train_class_proportions}")
            print(f"  Test: {test_class_proportions}")
            print(f"  Main Test: {main_test_class_proportions}")

            # Collect data for plotting
            for class_label, freq in train_class_counts.items():
                data.append({'client': c_id, 'class': class_label, 'frequency': freq, 'dataset': 'train'})
            #for class_label, freq in test_class_counts.items():
            #    data.append({'client': c_id, 'class': class_label, 'frequency': freq, 'dataset': 'test'})
            #for class_label, freq in main_test_class_counts.items():
            #    data.append({'client': c_id, 'class': class_label, 'frequency': freq, 'dataset': 'main_test'})
            
            print(f"Client {c_id} class distribution:")
            print(f"  Train: {train_class_counts}")
            #print(f"  Test: {test_class_counts}")
            #print(f"  Main Test: {main_test_class_counts}")
        print(f'generated {self.num_clients} clients with data')
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        # Plotting
        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=df, x='client', y='class', size='frequency', hue='dataset', sizes=(20, 200), alpha=0.6, palette='muted')
        #sns.scatterplot(data=df, x='client', y='class')
        plt.title('Class Distribution per Client')
        plt.xlabel('Client')
        plt.ylabel('Class')
        plt.legend(title='Dataset')
        plt.grid(True)
        plt.savefig('class_distribution_plot.png')
        plt.show()


    def init_client_models_optims(self, input_channels=1):
        """
        Initialize client-side model splits: front, back, center.
        Splits are made from Resnet50/18, effnetv2_small,
        and optimizer for each split: Adam / AdamW / Novograd.
        """
        try:
            model = importlib.import_module(self.import_module)
        except ImportError as e:
            print(f"Error importing module: {e}")
            return
        
        pretrained = self.args.pretrained
        lr = self.args.client_lr

        for c_id, client in self.clients.items():
            client.device = self.device

            try:
                client.front_model = model.front().to(self.device)
                client.front_model.eval()
            except AttributeError as e:
                print(f"Error initializing front model for client {c_id}: {e}")
                continue

            try:
                client.back_model = model.back().to(self.device)
                #client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr,weight_decay=1e-5)
                client.back_optimizer = Adam(client.back_model.parameters(), lr=lr) #same as previous one
                #client.back_scheduler = CosineAnnealingWarmRestarts(
                #    client.back_optimizer,
                #    T_0=20,
                #    T_mult=1,
                #    eta_min=1e-10
                #)
            except AttributeError as e:
                print(f"Error initializing back model or optimizer for client {c_id}: {e}")
                continue

        print(f'Initialized client-side model splits front & back and their optimizers')
        
        
    def init_clients_server_copy(self):
        """
        For each client, there is a server copy of the center model
        initialized using class: ConnectedClient(client_id, connection).
        The center model and its optimizer are initialized as well.
        """
        try:
            model = importlib.import_module(self.import_module)
        except ImportError as e:
            print(f"Error importing module: {e}")
            return
        
        pretrained = self.args.pretrained
        lr = self.args.server_lr

        self.sc_clients = dict()
        for c_id in self.client_ids:
            self.sc_clients[c_id] = ConnectedClient(id=c_id, conn=None)

        for c_id, sc_client in self.sc_clients.items():
            sc_client.device = self.device

            try:
                sc_client.center_front_model = model.center_front().to(self.device)
                sc_client.center_front_model.eval()
            except AttributeError as e:
                print(f"Error initializing center front model for server copy of client {c_id}: {e}")
                continue
            
            try:
                discriminator = Discriminator().to(self.device)
                sc_client.discriminator = discriminator
                sc_client.discriminator_loss_fn = nn.MSELoss()
                sc_client.discriminator_optimizer = Adam(sc_client.discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))
            except AttributeError as e:
                print(f"Error initializing discriminator model for server copy of client {c_id}: {e}")
            
            try:
                sc_client.center_back_model = model.center_back().to(self.device)
                #s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)
                sc_client.center_optimizer = Adam(sc_client.center_back_model.parameters(), lr=lr)
                #sc_client.center_scheduler = CosineAnnealingWarmRestarts(
                #    sc_client.center_optimizer,
                #    T_0=20,
                #    T_mult=1,
                #    eta_min=1e-10
                #)
            except AttributeError as e:
                print(f"Error initializing center back model or optimizer for server copy of client {c_id}: {e}")
                continue

        print(f'Initialized server-side model splits center_front, center_back, and center_back optimizer')
    
    
    def _create_save_dir(self):
        """
        Create the directory for saving models. The directory is created based on
        the dataset name and model split specified in the arguments. If the directory
        already exists, it won't be recreated.

        Raises:
            OSError: If there is an error creating the directory.
        """
        try:
            self.save_dir = Path(f'./saved_models/{self.args.dataset}/key_value_mode/model_split{self.args.split}')
            self.save_dir.mkdir(exist_ok=True, parents=True)
            print(f"Directory created at {self.save_dir}")
        except OSError as e:
            print(f"Error creating directory {self.save_dir}: {e}")
            raise

    
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
            
    def merge_model_weights(self, epoch):
        """
        Merge weights and distribute over all server-side center_back models.
        In the personalization phase, merging of weights of the back model layers is stopped:
            - Merge weights and distribute over all client-side back models if not in personalization mode.

        Args:
            epoch (int): The current epoch during which merging is applied.
        """
        print(f'Merging model weights at epoch {epoch}')

        params = []
        sample_lens = []

        # Collect the state dictionaries and sample lengths for all server-side center_back models
        for c_id, sc_client in self.sc_clients.items():
            try:
                params.append(copy.deepcopy(sc_client.center_back_model.state_dict()))
                sample_lens.append(len(self.clients[c_id].train_dataset) * self.args.kv_factor)
            except Exception as e:
                print(f"Error collecting weights for server copy client {c_id}: {e}")
                continue

        try:
            # Merge weights using a custom utility function
            w_glob = merge_weights(params, sample_lens)

            # Distribute the merged weights to all server-side center_back models
            for c_id, sc_client in self.sc_clients.items():
                sc_client.center_back_model.load_state_dict(w_glob)
                #print(f"Merged weights loaded to server copy client {c_id}")
        except Exception as e:
            print(f"Error merging or distributing server-side weights: {e}")

        if True:
            params = []
            sample_lens = []

            # Collect the state dictionaries and sample lengths for all client-side back models
            for c_id, client in self.clients.items():
                try:
                    params.append(copy.deepcopy(client.back_model.state_dict()))
                    sample_lens.append(len(client.train_dataset))
                except Exception as e:
                    print(f"Error collecting weights for client {c_id}: {e}")
                    continue

            try:
                # Merge weights using a custom utility function
                w_glob_cb = merge_weights(params, sample_lens)
        
                # Distribute the merged weights to all client-side back models
                for c_id, client in self.clients.items():
                    client.back_model.load_state_dict(w_glob_cb)
                    #print(f"Merged weights loaded to client {c_id}")
            except Exception as e:
                print(f"Error merging or distributing client-side weights: {e}")

        # Clean up to free memory
        del params, sample_lens

    
    def create_iters(self, dl='train'):
        """
        - Append 0 for train_f1/test_f1 per client list.
        - Assign iterators per client from train/test dataloader.
        """
        num_iters = {c_id: 0 for c_id in self.client_ids}
        for c_id, client in self.clients.items():
            if dl == 'train':
                client.train_f1.append(0)
                client.flag = False
            elif dl == 'test':
                client.test_f1.append(0)
                #client.test_iterator = iter(client.test_DataLoader)
                #client.num_test_iterations = len(client.test_DataLoader)
                #num_iters[c_id] = len(client.test_DataLoader)
        print(f"Created list to store f1 score for {dl} data.")
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
                client.kv_flag=1
                self.sc_clients[c_id].kv_flag=1
                client.num_iterations = len(client.train_DataLoader)
                client.iterator = iter(client.train_DataLoader)
            # forward client-side front model which sets activation and target mappings.
            for c_id, client in tqdm(self.clients.items()):
                #print(c_id,client.num_iterations)
                for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="client front"):
                    client.forward_front_key_value()
                    self.sc_clients[c_id].remote_activations1 = client.remote_activations1
                    self.sc_clients[c_id].batchkeys = client.key
                    self.sc_clients[c_id].forward_center_front()
                print(f"Training Set Key Value Store Created for Client {c_id}")
                print("Training Set Key Value Store Length is :", len((list(self.sc_clients[c_id].activation_mappings.keys()))))
                client.kv_flag=0
                self.sc_clients[c_id].kv_flag=0
        else:
            # create iterators for initial forward pass of testing phase
            for c_id, client in self.clients.items():
                client.kv_test_flag=1
                self.sc_clients[c_id].kv_test_flag=1
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                
            # forward client-side front model which sets activation and target mappings.
            for c_id, client in tqdm(self.clients.items()):
                #print(c_id,client.num_test_iterations)
                for it in tqdm(range(client.num_test_iterations * self.args.kv_factor),desc="server front"):
                    #if client.data_key % len(client.train_dataset) == 0:
                    client.forward_front_key_value_test()
                    self.sc_clients[c_id].remote_activations1 = client.remote_activations1
                    self.sc_clients[c_id].test_batchkeys = client.test_key
                    self.sc_clients[c_id].forward_center_front_test()
                print(f"Validation Set Key Value Store Created for Client {c_id}")
                print("Validation Set Key Value Store Length is :", len(list(self.sc_clients[c_id].test_activation_mappings.keys())))
                client.kv_test_flag=0
                self.sc_clients[c_id].kv_test_flag=0

    def populate_key_value_store(self,):
        """
        - resets key-value store for client and server
        - populates key-value store kv_factor no. of times
        """
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

        print(f"\n\nGeneralisation Phase Training {epoch}..........................................................................................")
        num_iters = self.create_iters(dl='train')
        self.overall_f1['train'].append(0)
        self.overall_acc['train'].append(0)
                
        for client_id, client in tqdm(self.clients.items()):
                client.iterator = iter(client.train_DataLoader)
                client.num_iterations = len(client.train_DataLoader)
                #for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="Training"):
                for iteration in tqdm(range(client.num_iterations),desc="Generalization Phase Training"):
                    client.forward_front_key_value()
                    self.sc_clients[client_id].batchkeys = client.key
                    
                    self.sc_clients[client_id].forward_center_front()
                    self.sc_clients[client_id].forward_center_back()
                    client.remote_activations2 = self.sc_clients[client_id].remote_activations2
                    
                    client.forward_back()
                    client.calculate_loss(mode='train')
                    
                    wandb.log({'train step loss': client.loss.item()})
                    
                    client.loss.backward()
                    
                    self.sc_clients[client_id].remote_activations2 = client.remote_activations2
                    self.sc_clients[client_id].backward_center()
                    
                    client.step_back()
                    #client.back_scheduler.step()
                    client.zero_grad_back()
                    
                    self.sc_clients[client_id].center_optimizer.step()
                    self.sc_clients[client_id].center_optimizer.zero_grad()
                    
                    f1=client.calculate_train_metric()
                    client.train_f1[-1] += f1 
                    
                    #print("train f1 per iteration: ",iteration,f1)
                    wandb.log({f'train f1 / iter: client {client_id}':f1.item()})

        # calculate per epoch metrics
        bal_accs, f1_macros = [], []
        avg_loss = 0
        for c_id, client in self.clients.items():
            client.train_f1[-1] /= client.num_iterations
            client.train_loss /= client.num_iterations
            avg_loss += client.train_loss
            self.overall_f1['train'][-1] += client.train_f1[-1]
            bal_acc_client, f1_macro_client = client.get_main_metric(mode='train') 
            bal_accs.append(bal_acc_client)
            f1_macros.append(f1_macro_client)
            wandb.log({f'train f1 {c_id}': client.train_f1[-1].item()})
            wandb.log({f'train accuracy {c_id}':bal_acc_client})
            wandb.log({f'tarin f1 macro {c_id}':f1_macro_client})
            wandb.log({f'train loss {c_id}': client.train_loss})
            client.train_loss = 0 # reset for next epoch


        # calculate per epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc['train'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1['train'][-1] /= self.num_clients
        print("avg train f1 all clients: ", self.overall_f1['train'][-1].item())
        print("avg train accuracy all clients: ", self.overall_acc['train'][-1])
        wandb.log({'avg train f1 all clients': self.overall_f1['train'][-1].item()})
        wandb.log({'avg train bal acc all clients': bal_acc})
        wandb.log({'avg train f1 macro all clients': f1_macro})
        wandb.log({'avg train loss all clients': avg_loss / self.num_clients})
        self.merge_model_weights(epoch)
        #if not self.pooling_mode:
            # merge model weights (center and back)
            #print()
            #self.merge_model_weights(epoch)
    
    def train_one_epoch_discriminator(self, epoch):
        '''
        In this epoch:
            -for every batch of data available:
                - forward the self.middle_activations to the auto encoder for training
                - back propagate the loss gradient
                - step the optimizer
                - calculate the loss metric
        '''
        print(f"\n\n Discriminator Phase Training {epoch}..........................................................................................")
        
        for client_id, client in tqdm(self.clients.items()):
            client.iterator = iter(client.train_DataLoader)
            client.num_iterations = len(client.train_DataLoader)
            
            for iteration in tqdm(range(client.num_iterations), desc="Discriminator Phase Training"):
                client.forward_front_key_value()
                
                self.sc_clients[client_id].batchkeys=client.key
                
                self.sc_clients[client_id].forward_center_front()
                self.sc_clients[client_id].forward_discriminator()
                self.sc_clients[client_id].calculate_discriminator_loss(mode="train")
                wandb.log({'discriminator step loss': self.sc_clients[client_id].disc_loss.item()})
                self.sc_clients[client_id].disc_loss.backward()
                self.sc_clients[client_id].discriminator_step()
                self.sc_clients[client_id].zero_grad_back()
            
        avg_loss = 0
            
        for c_id, client in self.clients.items():
            self.sc_clients[c_id].discriminator_train_loss /= client.num_iterations
            avg_loss+=self.sc_clients[c_id].discriminator_train_loss
            # wandb.log({f'train f1 {c_id}': client.train_f1[-1].item()})
            wandb.log({f'discriminator train loss of discriminator for {c_id}': self.sc_clients[c_id].discriminator_train_loss})
            self.sc_clients[c_id].discriminator_train_loss = 0
        
        wandb.log({f'avg discriminator train loss of clients': avg_loss / self.num_clients})       
    
    def train_one_epoch_personalise(self,epoch):
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
        print(f" \n\n Personalisation Phase Training {epoch}.........................................................................................")                
        for client_id, client in tqdm(self.clients.items()):
                client.iterator = iter(client.train_DataLoader)
                client.num_iterations = len(client.train_DataLoader)
                #for it in tqdm(range(client.num_iterations * self.args.kv_factor),desc="Training"):
                for iteration in tqdm(range(client.num_iterations),desc="Personlaisation Phase Training"):
                    client.forward_back_personalise()
                    client.calculate_loss(mode='train')
                    client.loss.backward()
                    wandb.log({'train step loss': client.loss.item()})
                    client.step_back()
                    client.zero_grad_back()
                    #client.loss.backward()
                    f1=client.calculate_train_metric()
                    client.train_f1[-1] += f1
                    #print("train f1 per iteration: ",iteration,f1)
                    wandb.log({f'train f1 / iter: client {client_id}':f1.item()})

        # calculate per epoch metrics
        bal_accs, f1_macros = [], []
        avg_loss = 0
        for c_id, client in self.clients.items():
            client.train_f1[-1] /= client.num_iterations
            client.train_loss /= client.num_iterations
            avg_loss += client.train_loss
            self.overall_f1['train'][-1] += client.train_f1[-1]
            bal_acc_client, f1_macro_client = client.get_main_metric(mode='train') 
            bal_accs.append(bal_acc_client)
            f1_macros.append(f1_macro_client)
            wandb.log({f'train f1 {c_id}': client.train_f1[-1].item()})
            wandb.log({f'train accuracy {c_id}':bal_acc_client})
            wandb.log({f'tarin f1 macro {c_id}':f1_macro_client})
            wandb.log({f'train loss {c_id}': client.train_loss})
            client.train_loss = 0 # reset for next epoch


        # calculate per epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc['train'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1['train'][-1] /= self.num_clients
        print("avg train f1 all clients: ", self.overall_f1['train'][-1].item())
        print("avg train accuracy all clients: ", self.overall_acc['train'][-1])
        wandb.log({'avg train f1 all clients': self.overall_f1['train'][-1].item()})
        wandb.log({'avg train bal acc all clients': bal_acc})
        wandb.log({'avg train f1 macro all clients': f1_macro})
        wandb.log({'avg train loss all clients': avg_loss / self.num_clients})
        
    
    @torch.no_grad()
    def test_one_epoch_personalise(self,epoch):
        """
        in this epoch:
            - for every batch of data available:
                - forward center_back model of server
                - forward back model of client
                - calculate batch metric

            - calculate epoch metric per client
            - calculate epoch metric avg. for all clients
        """

        for c_id, client in self.clients.items():
            client.pred = []
            client.y = []

        for client_id, client in tqdm(self.clients.items()):
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                for iteration in tqdm(range(client.num_test_iterations),desc="Personalised Validation"):
                    client.forward_back_personalise_test()
                    #client.forward_front_key_value_test()
                    client.calculate_loss(mode='test')
                    wandb.log({'Validation step loss': client.loss.item()})
                    f1=client.calculate_test_metric()
                    client.test_f1[-1] += f1 
                    #print("validation f1 per iteration: ",iteration,f1)
                    wandb.log({f'Validation f1 / iter: client {client_id}':f1.item()})
                    
        # calculate per epoch metrics
        avg_loss = 0
        bal_accs,f1_macros = [], []
        for c_id, client in self.clients.items():
            client.test_f1[-1] /= len(client.test_DataLoader)
            client.test_loss /= len(client.test_DataLoader)
            # calculate remaining metrics
            avg_loss += client.test_loss
            bal_acc_client, f1_macro_client = client.get_main_metric(mode='test')
            bal_accs.append(bal_acc_client)
            f1_macros.append(f1_macro_client)
            self.overall_f1['test'][-1] += client.test_f1[-1]
            wandb.log({f'Validation f1 {c_id}': client.test_f1[-1].item()})
            wandb.log({f'Validation accuracy {c_id}':bal_acc_client})
            wandb.log({f'Validation macro f1 {c_id}':f1_macro_client})
            wandb.log({f'Validation loss {c_id}': client.test_loss})
            client.test_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc['test'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1['test'][-1] /= self.num_clients
        print("validation f1: ", self.overall_f1['test'][-1])
        print("validation acc: ", self.overall_acc['test'][-1])
        wandb.log({'Validation avg f1 all clients': self.overall_f1['test'][-1].item()})
        wandb.log({'validation avg accuracy all clients': bal_acc})
        wandb.log({'Validation avg f1 macro all clients': bal_acc})
        wandb.log({'Validation avg loss all clients': avg_loss / self.num_clients}) 

    @torch.no_grad()
    def test_one_epoch_disc(self, epoch):
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
        #self.overall_f1['test'].append(0)
        #self.overall_acc['test'].append(0)

        #for c_id, client in self.clients.items():
        #    client.pred = []
        #    client.y = []

        for client_id, client in tqdm(self.clients.items()):
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                for iteration in tqdm(range(client.num_test_iterations),desc="Validation"):
                    client.forward_front_key_value_test()
                    self.sc_clients[client_id].test_batchkeys = client.test_key
                    self.sc_clients[client_id].forward_center_front_test()
                    # added by acs
                    self.sc_clients[client_id].forward_discriminator_test()
                    self.sc_clients[client_id].calculate_discriminator_loss(mode="test")
                    
                    wandb.log({'discriminator Validation step loss': self.sc_clients[client_id].disc_loss.item()})
                    
                    #self.sc_clients[client_id].forward_center_back()
                    #client.remote_activations2 = self.sc_clients[client_id].remote_activations2
                    #client.forward_back()
                    #client.calculate_loss(mode='test')
                    #wandb.log({'Validation step loss': client.loss.item()})
                    #f1=client.calculate_test_metric()
                    #client.test_f1[-1] += f1 
                    #print("validation f1 per iteration: ",iteration,f1)
                    #wandb.log({f'Validation f1 / iter: client {client_id}':f1.item()})
                    
        # calculate per epoch metrics
        avg_disc_test_loss = 0
        #bal_accs,f1_macros = [], []
        for c_id, client in self.clients.items():
            #client.test_f1[-1] /= len(client.test_DataLoader)
            #client.test_loss /= len(client.test_DataLoader)
            # added by acs
            self.sc_clients[c_id].discriminator_test_loss /= len(client.test_DataLoader)
            # calculate remaining metrics
            avg_disc_test_loss += self.sc_clients[c_id].discriminator_test_loss
            #bal_acc_client, f1_macro_client = client.get_main_metric(mode='test')
            #bal_accs.append(bal_acc_client)
            #f1_macros.append(f1_macro_client)
            #self.overall_f1['test'][-1] += client.test_f1[-1]
            #wandb.log({f'Validation f1 {c_id}': client.test_f1[-1].item()})
            #wandb.log({f'Validation accuracy {c_id}':bal_acc_client})
            #wandb.log({f'Validation macro f1 {c_id}':f1_macro_client})
            #wandb.log({f'Validation loss {c_id}': client.test_loss})
            #added by acs
            wandb.log({f'Validation loss of {c_id} discriminator': self.sc_clients[c_id].discriminator_test_loss})
            self.clients_threshold[c_id]=self.sc_clients[c_id].discriminator_test_loss
            self.sc_clients[c_id].discriminator_test_loss=0
            
            #client.test_loss = 0 # reset for next epoch
        print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Thresholds : {self.clients_threshold}')  
        # calculate epoch metrics across clients
        # bal_acc = np.array(bal_accs).mean()
        # self.overall_acc['test'][-1]=bal_acc
        # f1_macro = np.array(f1_macros).mean()
        # self.overall_f1['test'][-1] /= self.num_clients
        # print("validation f1: ", self.overall_f1['test'][-1])
        # print("validation acc: ", self.overall_acc['test'][-1])
        # wandb.log({'Validation avg f1 all clients': self.overall_f1['test'][-1].item()})
        # wandb.log({'validation avg accuracy all clients': bal_acc})
        # wandb.log({'Validation avg f1 macro all clients': bal_acc})
        wandb.log({'Discriminator validation avg loss of all clients': avg_disc_test_loss / self.num_clients}) 
        # if self.overall_acc['test'][-1] > self.best_acc:
        #     print(self.best_acc)
        #     self.best_acc = self.overall_acc['test'][-1]
        #     self.best_epoch = epoch
        #     self.early_stop_counter = 0
        #     print(f"MAX Validation Accuracy Score: {self.best_acc} @ epoch {self.best_epoch}")
        #     wandb.log({
        #         'max validation accuracy score':self.best_acc,
        #         'max_validation_accuarcy_epoch':self.best_epoch
        #     })
        #     return True
        # else:
        #     self.early_stop_counter += 1
        #     return False
    
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
        self.overall_f1['test'].append(0)
        self.overall_acc['test'].append(0)

        for c_id, client in self.clients.items():
            client.pred = []
            client.y = []

        for client_id, client in tqdm(self.clients.items()):
                client.num_test_iterations = len(client.test_DataLoader)
                client.test_iterator = iter(client.test_DataLoader)
                for iteration in tqdm(range(client.num_test_iterations),desc="Validation"):
                    client.forward_front_key_value_test()
                    self.sc_clients[client_id].test_batchkeys = client.test_key
                    self.sc_clients[client_id].forward_center_front_test()
                    self.sc_clients[client_id].forward_center_back()
                    client.remote_activations2 = self.sc_clients[client_id].remote_activations2
                    client.forward_back()
                    client.calculate_loss(mode='test')
                    wandb.log({'Validation step loss': client.loss.item()})
                    f1=client.calculate_test_metric()
                    client.test_f1[-1] += f1 
                    #print("validation f1 per iteration: ",iteration,f1)
                    wandb.log({f'Validation f1 / iter: client {client_id}':f1.item()})
                    
        # calculate per epoch metrics
        avg_loss = 0
        bal_accs,f1_macros = [], []
        for c_id, client in self.clients.items():
            client.test_f1[-1] /= len(client.test_DataLoader)
            client.test_loss /= len(client.test_DataLoader)
            # calculate remaining metrics
            avg_loss += client.test_loss
            bal_acc_client, f1_macro_client = client.get_main_metric(mode='test')
            bal_accs.append(bal_acc_client)
            f1_macros.append(f1_macro_client)
            self.overall_f1['test'][-1] += client.test_f1[-1]
            wandb.log({f'Validation f1 {c_id}': client.test_f1[-1].item()})
            wandb.log({f'Validation accuracy {c_id}':bal_acc_client})
            wandb.log({f'Validation macro f1 {c_id}':f1_macro_client})
            wandb.log({f'Validation loss {c_id}': client.test_loss})
            client.test_loss = 0 # reset for next epoch

        # calculate epoch metrics across clients
        bal_acc = np.array(bal_accs).mean()
        self.overall_acc['test'][-1]=bal_acc
        f1_macro = np.array(f1_macros).mean()
        self.overall_f1['test'][-1] /= self.num_clients
        print("validation f1: ", self.overall_f1['test'][-1])
        print("validation acc: ", self.overall_acc['test'][-1])
        wandb.log({'Validation avg f1 all clients': self.overall_f1['test'][-1].item()})
        wandb.log({'validation avg accuracy all clients': bal_acc})
        wandb.log({'Validation avg f1 macro all clients': bal_acc})
        wandb.log({'Validation avg loss all clients': avg_loss / self.num_clients}) 
        if self.overall_acc['test'][-1] > self.best_acc:
            print(self.best_acc)
            self.best_acc = self.overall_acc['test'][-1]
            self.best_epoch = epoch
            self.early_stop_counter = 0
            print(f"MAX Validation Accuracy Score: {self.best_acc} @ epoch {self.best_epoch}")
            wandb.log({
                'max validation accuracy score':self.best_acc,
                'max_validation_accuarcy_epoch':self.best_epoch
            })
            return True
        else:
            self.early_stop_counter += 1
            return False
                    
    def save_models(self,epoch):
        """
        save client-side back and server-side center_back models to disk
        """
        print("Save Model at epoch", epoch)
        if self.personalization_mode ==False:
            print("Save Best Model for Generalisation Phase")
            for c_id in self.client_ids:
                # client-side front model
                front_state_dict = self.clients[c_id].front_model.state_dict()
                torch.save(front_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_front.pth')
                torch.save(self.clients[c_id].front_model, self.save_dir / f'client_{c_id}_{self.args.model}_front_model.pth')
                # server-side center_front model
                center_front_state_dict = self.sc_clients[c_id].center_front_model.state_dict()
                torch.save(center_front_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_center_front.pth')
                torch.save(self.sc_clients[c_id].center_front_model, self.save_dir / f'client_{c_id}_{self.args.model}_center_front_model.pth')
                # server-side center_back model
                center_back_state_dict = self.sc_clients[c_id].center_back_model.state_dict()
                torch.save(center_back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_center_back.pth')
                torch.save(self.sc_clients[c_id].center_back_model, self.save_dir / f'client_{c_id}_{self.args.model}_center_back_model.pth')
                # client-side back model
                back_state_dict = self.clients[c_id].back_model.state_dict()
                torch.save(back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_back.pth')
                torch.save(self.clients[c_id].back_model, self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth')
        else:
            for c_id in self.client_ids:
                print("Save Best Model for Personlalisation Phase")
                # client-side back model
                back_state_dict = self.clients[c_id].back_model.state_dict()
                torch.save(back_state_dict, self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth')
                torch.save(self.clients[c_id].back_model, self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth')
            
    def load_best_models(self,):
        """
        replaces the latest models with the best models on server and client-side
        """
        print("Loaded Best Model")

        model = importlib.import_module(self.import_module)

        for c_id in self.client_ids:
            front = model.front().to(self.device)
            front_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_front.pth')
            front.load_state_dict(front_sd)
            # front = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_front_model.pth').to(self.device)
            center_front = model.center_front().to(self.device)
            center_front_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_front.pth')
            center_front.load_state_dict(center_front_sd)
            # center_front = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_front_model.pth').to(self.device)
            center_back = model.center_back().to(self.device)
            center_back_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_back.pth')
            center_back.load_state_dict(center_back_sd)
            # center_back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_back_model.pth').to(self.device)
            back = model.back().to(self.device)
            if self.personalization_mode == False:
                print("Load Best Model for Generalisation Phase")
                back_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back.pth')
                # back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth').to(self.device)
            else:
                print("Load Best Model for Personalisation Phase")
                back_sd = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per.pth')
                # back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth').to(self.device)
            back.load_state_dict(back_sd)
            
            self.clients[c_id].front_model = front
            self.clients[c_id].back_model = back
            self.sc_clients[c_id].center_front_model = center_front
            self.sc_clients[c_id].center_back_model = center_back
            
            self.clients[c_id].front_model.eval()
            self.clients[c_id].back_model.eval()
            self.sc_clients[c_id].center_front_model.eval()
            self.sc_clients[c_id].center_back_model.eval()

    @torch.no_grad()
    def inference(self,):
        """
        run inference on the main test dataset
        """

        print("RUNNING INFERENCE from the best models on test dataset")

        # self.load_best_models()
        avg_acc=0
        for c_id in tqdm(self.client_ids,desc="Testing"):
            # front = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_front_model.pth').to(self.device)
            # front.eval()
            # center_front = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_front_model.pth').to(self.device)
            # center_front.eval()
            # center_back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_center_back_model.pth').to(self.device)
            # center_back.eval()
            # if self.personalization_mode == False:
            #     print("Load Best Model for Generalisation Phase")
            #     back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth').to(self.device)
            # else:
            #     print("Load Best Model for Personalisation Phase")
            #     back = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth').to(self.device)
            # back.eval()
            trues = []
            preds = []

            for batch in self.clients[c_id].main_test_DataLoader:
                image, label = batch['image'].to(self.device), batch['label'].to(self.device)

                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4 = self.clients[c_id].back_model(x3)
                # x1 = front(image)
                # x2 = center_front(x1)
                # x3 = center_back(x2)
                # x4 = back(x3)

                trues.append(label.cpu())
                preds.append(x4.cpu())

            preds = torch.cat(preds)
            targets = torch.cat(trues)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            #bacc = balanced_accuracy_score(targets, preds)
            wandb.log({
                'inference cfm': wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=targets,
                    class_names=[f'{i}' for i in range(10)]
                )
            })

            print(f'inference score {c_id}: {accuracy}')
            avg_acc+=accuracy
            wandb.log({f'inference score {c_id}': accuracy})
        print(f'Average inference score: {avg_acc/len(self.clients)}')

    def inference_new(self,):
        '''
        run inference individually on every data point from main_test_dataset
        '''
        print('running inference_new on main_test_dataset')
        avg_acc=0
        # for c_id in tqdm(self.client_ids,desc="Testing_New"):
        
        # self.load_best_models()
        for idx, (c_id, client) in enumerate(self.clients.items()):
            trues=[]
            preds=[]
            generalized=[]
            personalized=[]
            c=0
            c1=0
            c2=0
            for data in client.main_test_dataset:
                c+=1
                self.sc_clients[c_id].discriminator.eval()
                image = data['image'].to(self.device)
                image = torch.unsqueeze(image, 0)
                label = data['label'].to(self.device)
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                
                reconstruction = self.sc_clients[c_id].discriminator(x2)
                loss = self.sc_clients[c_id].discriminator_loss_fn(reconstruction, x2)
                
                if loss > self.clients_threshold[c_id]:
                    c1+=1
                    # print(f"{loss}>{self.clients_threshold[c_id]}")
                    generalized.append([image, label])
                else:
                    c2+=1
                    # print(f"less")
                    personalized.append([image, label])
            print(f'test images in  {c_id} : {c} ; {c1},{c2}')
            
            print(f"------------no of data points predicted ood in {c_id}: {len(generalized)}")
            self.clients[c_id].back_model = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_model.pth')
            for image, label in generalized:
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4 = self.clients[c_id].back_model(x3)
                
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            
            print(f"------------no of data points predicted id in {c_id}: {len(personalized)}")
            self.clients[c_id].back_model = torch.load(self.save_dir / f'client_{c_id}_{self.args.model}_back_per_model.pth')
            for image, label in personalized:
                x1 = self.clients[c_id].front_model(image)
                x2 = self.sc_clients[c_id].center_front_model(x1)
                x3 = self.sc_clients[c_id].center_back_model(x2)
                x4 = self.clients[c_id].back_model(x3)
                
                trues.append(label.cpu())
                preds.append(x4.cpu())
            
            trues= [torch.unsqueeze(t, 0) for t in trues]
            preds = torch.cat(preds)
            targets = torch.cat(trues)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            targets = targets.reshape(-1).numpy()
            preds = np.argmax(preds.detach().numpy(), axis=1)
            correct = np.sum(preds == targets)
            total = len(targets)
            accuracy = correct / total
            
            wandb.log({
                'inference cfm': wandb.plot.confusion_matrix(
                    preds=preds,
                    y_true=targets,
                    class_names=[f'{i}' for i in range(10)]
                )
            })
            
            avg_acc+=accuracy
            wandb.log({f'inference score {c_id}': accuracy})
        print(f'Average inference score: {avg_acc/len(self.clients)}')
    
    def clear_cache(self,):
        gc.collect()
        torch.cuda.empty_cache()
        
    def save_kv(self,):
            for c_id in tqdm(self.client_ids,desc="Client Side KV for Training"):
                for batch in self.clients[c_id].train_DataLoader:
                    image, label, batchkeys = batch['image'].to(self.device), batch['label'].to(self.device), batch['id']
                    #print(batchkeys)
                    #x1 = self.clients[c_id].front_model(image)
                    #x2 = self.sc_clients[c_id].center_front_model(x1)
                    valid_keys = [key for key in batchkeys if key in self.sc_clients[c_id].activation_mappings]
                    #print(valid_keys)
                    activations_list = [self.sc_clients[c_id].activation_mappings[key] for key in valid_keys]
                    activations_array = np.array(activations_list)
                    x2 = torch.tensor(activations_array, device=self.device)
                    x3 = self.sc_clients[c_id].center_back_model(x2)
                    local_middle_activations=list(x3.cpu().detach().numpy())
                    for i in range(0, len(batchkeys)):
                        #print(batchkeys[i])
                        self.clients[c_id].activation_mappings[batchkeys[i]]=local_middle_activations[i]
                print(f"Training Set Key Value Store Created for Client {c_id}")
                print("Training Set Key Value Store Length is :", len((list(self.clients[c_id].activation_mappings.keys()))))
                
                
            for c_id in tqdm(self.client_ids,desc="Client Side KV for Testing"):
                for batch in self.clients[c_id].test_DataLoader:
                    image, label, batchkeys = batch['image'].to(self.device), batch['label'].to(self.device), batch['id']
                    #x1 = self.clients[c_id].front_model(image)
                    #x2 = self.sc_clients[c_id].center_front_model(x1)
                    valid_keys = [key for key in batchkeys if key in self.sc_clients[c_id].test_activation_mappings]
                    activations_list = [self.sc_clients[c_id].test_activation_mappings[key] for key in valid_keys]
                    activations_array = np.array(activations_list)
                    x2 = torch.tensor(activations_array, device=self.device)
                    x3 = self.sc_clients[c_id].center_back_model(x2)
                    local_middle_activations=list(x3.cpu().detach().numpy())
                    for i in range(0, len(batchkeys)):
                        #print(batchkeys[i])
                        self.clients[c_id].activation_mappings[batchkeys[i]]=local_middle_activations[i]
                print(f"Validation Set Key Value Store Created for Client {c_id}")
                print("Validation Set Key Value Store Length is :", len((list(self.clients[c_id].activation_mappings.keys()))))


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
            
            if self.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

            wandb.log({'epoch':epoch})

            for c_id in self.client_ids:
                self.clients[c_id].back_model.train()
                self.sc_clients[c_id].center_back_model.train()

            self.train_one_epoch(epoch)
            self.clear_cache()

            for c_id in self.client_ids:
                self.clients[c_id].back_model.eval()
                self.sc_clients[c_id].center_back_model.eval()
                
            should_save = self.test_one_epoch(epoch)
            print(should_save)
            if should_save:
                self.early_stop = False
                print("Model improved and saved.")
                self.save_models(epoch)
                self.inference()
                #self.save_kv() # for personalisation phase
            else:
                if self.early_stop_counter == 5:
                    self.early_stop = True
    
            self.clear_cache()
        
        for run in range(60):
            for c_id in self.client_ids:
                self.sc_clients[c_id].discriminator.train()
            self.train_one_epoch_discriminator(run)  # added by acs
            for c_id in self.client_ids:
                self.sc_clients[c_id].discriminator.eval()
            self.test_one_epoch_disc(run)
                    
        #self.load_best_models
        #self.inference()
        # final metrics
        #print(f'\n\n\n{"::"*10}BEST METRICS{"::"*10}')
        #print("Training Mean f1 Score: ", self.overall_f1['train'][self.max_f1['epoch']])
        #print("Maximum Test Mean f1 Score: ", self.max_f1['f1'])
        self.inference()
        if self.personalize:
            self.personalization_mode = True
            self.personalize(epoch)
            print("Personalization Started")
            self.save_kv()
            for epoch in tqdm(range(epoch,self.args.epochs)):
                for c_id in self.client_ids:
                    self.clients[c_id].back_model.train()
                    self.sc_clients[c_id].center_back_model.eval()
                self.train_one_epoch_personalise(epoch)
                self.clear_cache()
                for c_id in self.client_ids:
                    self.clients[c_id].back_model.eval()
                self.test_one_epoch_personalise(epoch)
                self.clear_cache()
                self.save_models(epoch)
                self.inference()
        print("**HYBRID INFERENCE**")
        self.inference_new()
        
    def __init__(self,args):
        """
        implementation of PFSL training & testing simulation on ISIC-2019 dataset
            - ISIC-2019: Dermoscopy Image Classification
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

        self.import_module = f"ImageClassification_Task.models.{self.args.model}_split{self.args.split}"

        self.pooling_mode = self.args.pool

        # refresh key-value store every N epochs
        self.kv_refresh_rate = self.args.kv_refresh_rate

        wandb.login(key=WANDB_KEY)
        self.run = wandb.init(
            project='med-fsl_isic2019',
            config=vars(self.args),
            job_type='train',
            mode='online' if self.log_wandb else 'disabled'
        )

        self.seed()

        #self.isic = ISICDataBuilder()
        self.cifar_builder = CIFAR10DataBuilder()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.overall_f1 = {
            'train': [],
            'test': []
        }
        
        self.overall_acc = {
            'train': [],
            'test': []
        }

        self.max_f1 = {
            'f1': 0,
            'epoch': -1
        }
        
        self.max_acc = {
            'acc': 0,
            'epoch': -1
        }
        
        self.patience = 5
        #self.best_val_acc = 0
        self.early_stop_counter = 0
        self.early_stop = False
        self.best_acc = 0
        self.best_epoch = 0

        self.train_batch_size = self.args.batch_size
        self.test_batch_size = self.args.test_batch_size

        self.personalization_mode = False

        self.init_clients_with_data()

        self.init_client_models_optims()

        self.init_clients_server_copy()



if __name__ == '__main__':
    args = parse_arguments()
    trainer = ISICTrainer(args)
    trainer.fit()
    trainer.inference()