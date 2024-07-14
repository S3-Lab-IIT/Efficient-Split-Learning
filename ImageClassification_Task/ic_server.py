from threading import Thread
from utils.connections import is_socket_closed
from utils.connections import send_object
from utils.connections import get_object
import pickle
import queue
import struct
import torch
import hashlib
from tqdm.auto import tqdm
import numpy as np


def handle(client, addr, file):
    buffsize = 1024
    # file = '/home/ashutosh/score_report.pdf'
    # print('File size:', os.path.getsize(file))
    fsize = struct.pack('!I', len(file))
    print('Len of file size struct:', len(fsize))
    client.send(fsize)
    with open(file, 'rb') as fd:
        while True:
            chunk = fd.read(buffsize)
            if not chunk:
                break
            client.send(chunk)
        fd.seek(0)
        hash = hashlib.sha512()
        while True:
            chunk = fd.read(buffsize)
            if not chunk:
                break
            hash.update(chunk)
        client.send(hash.digest())


class ConnectedClient(object):
    # def __init__(self, id, conn, address, loop_time=1/60, *args, **kwargs):
    def __init__(self, id, conn, *args, **kwargs):
        super(ConnectedClient, self).__init__(*args, **kwargs)
        self.id = id
        self.conn = conn
        self.front_model = None
        self.back_model = None
        self.center_model = None
        self.center_front_model=None
        self.discriminator=None # added by acs
        self.discriminator_loss_fn = None # added by acs
        self.discriminator_train_loss = 0 # added by acs
        self.discriminator_test_loss = 0 # added by acs
        self.discriminator_main_test_loss = 0 # added by acs
        self.disc_loss = None #added by acs
        self.discriminator_optimizer = None # added by acs
        self.center_back_model=None
        self.train_fun = None
        self.test_fun = None
        self.keepRunning = True

        self.all_keys=[]
        self.current_keys=[]
        self.batchkeys = None
        self.test_batchkeys = None
        self.kv_flag=0
        self.kv_test_flag=0

        self.a1 = None
        self.a2 = None
        self.flag=0
        self.center_optimizer = None
        self.center_scheduler = None
        self.activation_mappings={}
        self.test_activation_mappings={}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # def onThread(self, function, *args, **kwargs):
    #     self.q.put((function, args, kwargs))


    # def run(self, loop_time=1/60, *args, **kwargs):
    #     super(ConnectedClient, self).run(*args, **kwargs)
    #     while True:
    #         try:
    #             function, args, kwargs = self.q.get(timeout=self.timeout)
    #             function(*args, **kwargs)
    #         except queue.Empty:
    #             self.idle()

    
    def forward_center(self):
        activations2 = self.center_front_model(self.remote_activations1)
        self.activations2 = self.center_back_model(activations2)
        self.remote_activations2 = self.activations2.detach().requires_grad_(True)

    def forward_center_front(self):
        if self.kv_flag==1:
            #print("Size of remote_activations1:", self.remote_activations1.size())
            self.middle_activations=self.center_front_model(self.remote_activations1)
            #print("Size of middle_activations:", self.middle_activations.size())
            local_middle_activations=list(self.middle_activations.cpu().detach().numpy())
            #print("Length of batchkey",len(self.batchkeys))
            for i in range(0, len(self.batchkeys)):
                #print(self.batchkeys[i])
                self.activation_mappings[self.batchkeys[i]]=local_middle_activations[i]
            
        else:
            # Ensure all keys in batchkeys exist in activation_mappings
            missing_keys = [key for key in self.batchkeys if key not in self.activation_mappings]
            if missing_keys:
                print(f"Warning: Missing keys in activation_mappings: {missing_keys}")

            # Retrieve activations for batchkeys from activation_mappings
            valid_keys = [key for key in self.batchkeys if key in self.activation_mappings]
            if not valid_keys:
                print("Error: No valid keys found in activation_mappings.")
                return

            # Convert the list of numpy arrays to a single numpy array
            activations_list = [self.activation_mappings[key] for key in valid_keys]
            activations_array = np.array(activations_list)
            
            # Convert the numpy array to a tensor and move it to the appropriate device
            self.middle_activations = torch.tensor(activations_array, device=self.device)
            #print("Middle activations created from train activation_mappings based on batchkeys.")
    
    def forward_discriminator(self):
        self.reconstructions = self.discriminator(self.middle_activations)
        
    def forward_discriminator_test(self):
        self.reconstructions = self.discriminator(self.middle_activations)
        
    
    def calculate_discriminator_loss(self, mode):
        
        self.disc_loss  = self.discriminator_loss_fn(self.reconstructions, self.middle_activations)
        if mode=="train":
            self.discriminator_train_loss+=self.disc_loss.item()
        elif mode=="test":
            self.discriminator_test_loss+=self.disc_loss.item()
        else:
            self.discriminator_main_test_loss+=self.disc_loss.item()
        
    def discriminator_step(self):
        self.discriminator_optimizer.step()
    
    def zero_grad_back(self):
        self.discriminator_optimizer.zero_grad()
            
    def forward_center_front_test(self):
        if self.kv_test_flag==1:
            #print("Size of remote_activations1:", self.remote_activations1.size())
            self.middle_activations=self.center_front_model(self.remote_activations1)
            #print("Size of middle_activations:", self.middle_activations.size())
            local_middle_activations=list(self.middle_activations.cpu().detach().numpy())
            #print("Length of batchkey",len(self.test_batchkeys))
            for i in range(0, len(self.test_batchkeys)):
                #print(self.test_batchkeys[i])
                self.test_activation_mappings[self.test_batchkeys[i]]=local_middle_activations[i]
            
        else:
            # Ensure all keys in batchkeys exist in activation_mappings
            missing_test_keys = [key for key in self.test_batchkeys if key not in self.test_activation_mappings]
            if missing_test_keys:
                print(f"Warning: Missing keys in activation_mappings: {missing_test_keys}")

            # Retrieve activations for batchkeys from activation_mappings
            valid_test_keys = [key for key in self.test_batchkeys if key in self.test_activation_mappings]
            if not valid_test_keys:
                print("Error: No valid keys found in activation_mappings.")
                return
            #print("valid keys", valid_test_keys)
            # Convert the list of numpy arrays to a single numpy array
            activations_test_list = [self.test_activation_mappings[key] for key in valid_test_keys]
            activations_test_array = np.array(activations_test_list)
            
            # Convert the numpy array to a tensor and move it to the appropriate device
            self.middle_activations = torch.tensor(activations_test_array, device=self.device)
            #print("Size of middle_activations:", self.middle_activations.size())
            #print("Middle activations created from validation activation_mappings based on batchkeys.")

    def forward_center_front_test_old(self):
        self.middle_activations=self.center_front_model(self.remote_activations1)
        local_middle_activations=list(self.middle_activations.cpu().detach().numpy())
        for i in range(0, len(self.current_keys)):
            self.test_activation_mappings[self.current_keys[i]]=local_middle_activations[i]
            


    def forward_center_back(self):
        self.activations2=self.center_back_model(self.middle_activations)
        #print("Size of activations2:", self.activations2.size())
        self.remote_activations2=self.activations2.detach().requires_grad_(True)
        

    def backward_center(self):
        self.activations2.backward(self.remote_activations2.grad)

    def update_all_keys(self):
        self.all_keys=list(set(self.all_keys)-set(self.current_keys))


    def idle(self):
        pass


    def connect(self):
        pass


    def disconnect(self):
        if not is_socket_closed(self.conn):
            self.conn.close()
            return True
        else:
            return False


    # def _send_model(self):
    def send_model(self):
        model = {'front': self.front_model, 'back': self.back_model}
        send_object(self.conn, model)
        # handle(self.conn, self.address, model)


    # def send_optimizers(self):
    #     # This is just a sample code and NOT optimizers. Need to write code for initializing optimizers
    #     optimizers = {'front': self.front_model.parameters(), 'back': self.back_model.parameters()}
    #     send_object(self.conn, optimizers)


    def send_activations(self, activations):
        send_object(self.conn, activations)


    def get_remote_activations1(self):
        self.remote_activations1 = get_object(self.conn)


    def send_remote_activations2(self):
        send_object(self.conn, self.remote_activations2)


    def get_remote_activations2_grads(self):
        self.remote_activations2.grad = get_object(self.conn)


    def send_remote_activations1_grads(self):
        send_object(self.conn, self.remote_activations1.grad)

    # def send_model(self):
    #     self.onThread(self._send_model)        
