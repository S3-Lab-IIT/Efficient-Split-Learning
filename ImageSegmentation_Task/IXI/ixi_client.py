import os
import torch
import torch.nn.functional as F
import multiprocessing
from threading import Thread
from utils.connections import is_socket_closed
from utils.connections import send_object
from utils.connections import get_object
import pickle
import queue
import struct
import numpy as np
import torchmetrics

# monai imports
from monai.data import DataLoader
from monai.losses import DiceLoss



class Client(Thread):
    def __init__(self, id, *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        self.id = id
        self.flag=0
        self.test_flag=0
        self.front_model = []
        self.skips = []
        self.back_model = []
        self.center_model = []
        self.losses = []


        self.current_keys=[]
        self.target_mappings={}
        self.skip_mappings={}
        self.activation_mappings={}
        self.data_key=0

        self.test_target_mappings={}
        self.test_skip_mappings={}
        self.test_activation_mappings={}
        self.test_data_key=0

        self.train_dataset = None
        self.test_dataset = None
        self.main_test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.main_test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.iterator = None
        self.test_iterator=None
        self.activations1 = None
        self.remote_activations1 = None
        self.outputs = None
        self.loss = None

        self.train_loss = 0
        self.test_loss = 0

        self.criterion = None
        self.data = None
        self.targets = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.back_scheduler = None
        self.train_dice = []
        self.test_dice = []
        self.front_epsilons = []
        self.front_best_alphas = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       


    @torch.no_grad()
    def compute_dice(self,preds,targets):
        dice = torchmetrics.functional.dice(
            preds=preds,
            target=targets.long(),
            zero_division=1e-8,
            ignore_index=0, # ignore bg
            num_classes=2
        )
        return dice


    def backward_back(self):
        self.loss.backward()

    def set_targets(self):
        self.targets=torch.Tensor(np.array([self.target_mappings[x] for x in self.current_keys])).to(self.device)

    def set_test_targets(self):
        self.targets=torch.Tensor(np.array([self.test_target_mappings[x] for x in self.current_keys])).to(self.device)
           
    def backward_front(self):
        self.activations1.backward(self.remote_activations1.grad)

    

    def calculate_loss(self, mode='train'):
        """
        loss function to calculate loss for ixi
        dice loss, 2 classes: 0: bg, 1:brain
        """
        loss_fn=DiceLoss(to_onehot_y=True,softmax=True)
        self.loss=loss_fn(self.outputs, self.targets)

        if mode=='train':
            self.train_loss += self.loss.item()
        elif mode=='test':
            self.test_loss += self.loss.item()

    @torch.no_grad()
    def calculate_train_dice_kits(self):
        preds = self.outputs
        targets = self.targets
        dice_coeff = self.compute_dice(preds.cpu(),targets.cpu())
        return dice_coeff
    
    @torch.no_grad()
    def calculate_test_dice_kits(self):
        preds = self.outputs
        targets = self.targets
        self.pred.extend(preds.cpu().detach().numpy().tolist())
        self.y.extend(targets.cpu().detach().numpy().tolist())
        dice_coeff = self.compute_dice(preds.cpu(),targets.cpu())
        return dice_coeff



    def connect_server(self, host='localhost', port=8000, BUFFER_SIZE=4096):
        self.socket, self.server_socket = multiprocessing.Pipe()
        print(f"[*] Client {self.id} connecting to {host}")


    def create_DataLoader(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        # MONAI DATALOADERS
        self.train_DataLoader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.train_batch_size,
                                           shuffle=True,
                                           )
        self.test_DataLoader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.test_batch_size,
                                          )
        self.main_test_DataLoader = DataLoader(dataset=self.main_test_dataset,
                                               batch_size=self.test_batch_size,
                                              )

    def disconnect_server(self) -> bool:
        if not is_socket_closed(self.socket):
            self.socket.close()
            return True
        else:
            return False


    def forward_back(self):
        # self.back_model.to(self.device)
        self.outputs = self.back_model(self.remote_activations2)
        # print("---", self.outputs)
        # print("+++", self.outputs.shape)

    def forward_front(self):
        batch_data = next(self.iterator)
        self.data, self.targets = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        
        # self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        self.skips = self.front_model.skips
        self.remote_activations1 = self.activations1.detach().requires_grad_(True)

        # return self.activations1

    
    def forward_front_key_value(self):
        batch_data = next(self.iterator)
        self.data, self.targets = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        # self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        local_activations1=list(self.activations1.cpu().detach().numpy())
        local_targets=list(self.targets.cpu().detach().numpy())
        
        local_skips=[list(s.detach().cpu().numpy()) for s in self.front_model.skips]
        local_skips=list(zip(*local_skips))
        
        for i in range(0,len(local_targets)):
            self.activation_mappings[self.data_key]=local_activations1[i]
            self.skip_mappings[self.data_key]=local_skips[i]
            self.target_mappings[self.data_key]=local_targets[i]
            self.data_key+=1


    def forward_front_key_value_test(self):
        batch_data = next(self.test_iterator)
        self.data, self.targets = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        # self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        local_activations1=list(self.activations1.cpu().detach().numpy())
        local_targets=list(self.targets.cpu().detach().numpy())
        
        local_skips=[list(s.detach().cpu().numpy()) for s in self.front_model.skips]
        local_skips=list(zip(*local_skips))
        
        for i in range(0,len(local_targets)):
            self.test_activation_mappings[self.test_data_key]=local_activations1[i]
            self.test_skip_mappings[self.test_data_key]=local_skips[i]
            self.test_target_mappings[self.test_data_key]=local_targets[i]
            self.test_data_key+=1
    

        


    def get_model(self):
        model = get_object(self.socket)
        self.front_model = model['front']
        self.back_model = model['back']


    def get_remote_activations1_grads(self):
        self.remote_activations1.grad = get_object(self.socket)


    def get_remote_activations2(self):
        self.remote_activations2 = get_object(self.socket)


    def idle(self):
        pass


    def send_remote_activations1(self):
        send_object(self.socket, self.remote_activations1)
    

    def send_remote_activations2_grads(self):
        send_object(self.socket, self.remote_activations2.grad)


    def step_front(self):
        self.front_optimizer.step()
        

    def step_back(self):
        self.back_optimizer.step()


    def zero_grad_front(self):
        self.front_optimizer.zero_grad()
        

    def zero_grad_back(self):
        self.back_optimizer.zero_grad()

