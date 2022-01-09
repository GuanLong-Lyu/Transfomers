# This script is for
from time import time
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,input_shape,head):
        super(Attention,self).__init__()
        self.head = head
        self.input_shape = input_shape
        self.head_dims =  int(input_shape // head)

        self.query = nn.Linear(self.head_dims,self.head_dims)
        self.key = nn.Linear(self.head_dims,self.head_dims)
        self.value = nn.Linear(self.head_dims,self.head_dims)
        self.fc = nn.Linear(self.head_dims*self.head, self.input_shape) # ?

    def foward(self,query, key, value, mask = None):
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(batch_size, query_len, self.head, self.head_dims)
        key = key.reshape(batch_size, key_len, self.head, self.head_dims)
        value = value.reshape(batch_size, value_len, self.head, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Scaled Dot-Product Attention
        score = torch.einsum('bqhd,bkhd->bhqk', [query,key]) # Matmul
        score = score/(self.head_dims**(1/2)) # scale
        score = torch.softmax(score, dim=1) # softmax

        out = torch.einsum('bhqv,bvhd->bqhd',[score,value]) # matmul(score, value)
        out = out.reshape(batch_size,query_len, self.head*self.head_dims)
        out = self.fc(out) # Linear

        return out







