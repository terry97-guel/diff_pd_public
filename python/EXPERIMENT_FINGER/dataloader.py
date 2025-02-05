# %%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os 
import json
import random


class JsonDataset(Dataset):
    def __init__(self,data):
        motor_control = torch.tensor(data["motor_control"],dtype=torch.float32)
        position = torch.tensor(data["position"],dtype=torch.float32)
        
        assert len(motor_control) == len(position)
        
        self.motor_control = motor_control
        self.position = position.unsqueeze(-1)
        
        assert self.position.shape[-2:] == (3,1)
        
                
    def __len__(self):
        return len(self.motor_control)
    
    def __getitem__(self,idx):
        return dict(motor_control = self.motor_control[idx], position = self.position[idx])


    def get_std_mean(self):
        motor_control = self.motor_control
        position = self.position
        
        # average along joints
        axis_mean_position = torch.mean(position, dim=1)

        motor_std, motor_mean = torch.std_mean(motor_control, dim=0)
        pos_std, pos_mean = torch.std_mean(axis_mean_position, dim=0)


        motor_std, motor_mean, pos_std, pos_mean
        
        return (motor_std, motor_mean), (pos_std, pos_mean)
        

def get_int_from_ratio(length, ratio):
    return int(ratio*length)

def get_dataset(data_path, data_ratio=1.0):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]
    ext_data = data["ext"]
    
    return JsonDataset(train_data), JsonDataset(val_data), JsonDataset(test_data), JsonDataset(ext_data)


class Sampler():
    def __init__(self, batch_size, dataset:JsonDataset):
        self.dataset = dataset
        
        self.length = len(dataset)
        self.max_iter = self.length//batch_size + 1
        self.current_iter = 0
        
        self.indices = torch.chunk(torch.randperm(self.length), self.max_iter)
        self.keep_idx = torch.randint(0, self.length, [get_int_from_ratio(self.length, self.focus_ratio),])
        
        self.loss_restore = torch.zeros(self.length)
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iter:
            self.current_iter = 0
            self.indices = torch.chunk(torch.randperm(len(self)), self.max_iter)
            
            sort_idx = torch.argsort(self.loss_restore, descending=True)
            self.keep_idx = sort_idx[:int(len(sort_idx)*self.focus_ratio)]
            raise StopIteration()

        else:
            self.sel_idx = self.indices[self.current_iter]
            idx = torch.cat([self.sel_idx, self.keep_idx])
            self.current_iter += 1
            return self.dataset[idx]
    
    def sample_all(self):
        idx = torch.arange(self.length)    
        return self.dataset[idx]