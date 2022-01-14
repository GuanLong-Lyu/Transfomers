
from time import time
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,embed_size,head):
        super(SelfAttention,self).__init__()
        self.head = head
        self.embed_size = embed_size
        self.head_dims =  int(embed_size // head)

        self.query = nn.Linear(self.head_dims,self.head_dims)
        self.key = nn.Linear(self.head_dims,self.head_dims)
        self.value = nn.Linear(self.head_dims,self.head_dims)
        self.fc = nn.Linear(self.head_dims*self.head, self.embed_size) # ?

    def foward(self, value, key, query, mask = None):
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

class TransformerBlock(nn.Module):
    def __init__(self,embed_size,head, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size=embed_size, head=head)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

        def forward(self, value, key, query, mask):
            attention = self.attention(value, key, query, mask)

            x = self.dropout(self.norm1(attention + query))
            forward = self.feed_forward(x)

            out = self(self.dropout(self.norm2(forward + x)))

            return out


class Encoder(nn.Module):




