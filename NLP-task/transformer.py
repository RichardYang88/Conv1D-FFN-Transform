import torch.nn as nn
from torch.nn.functional import cross_entropy,softmax, relu
import numpy as np
import torch
from torch.utils import data
import utils
from torch.utils.data import DataLoader
import argparse
MAX_LEN = 11

# set the style of FFN, mlp_ffn or conv_ffn, if you want to use conv1d_ffn, please set k_size = 3, 5 or 7
# style = "mlp_ffn"
style = "conv1d_ffn"
k_size = 3 # kernel size of conv1d_ffn

class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(n_head * self.head_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.attention = None

    def forward(self,q,k,v,mask,training):
        # residual connect
        residual = q
        dim_per_head= self.head_dim
        num_heads = self.n_head
        batch_size = q.size(0)

        # linear projection
        key = self.wk(k)    # [n, step, num_heads * head_dim]
        value = self.wv(v)  # [n, step, num_heads * head_dim]
        query = self.wq(q)  # [n, step, num_heads * head_dim]

        # split by head
        query = self.split_heads(query)       # [n, n_head, q_step, h_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(query,key, value, mask)    # [n, n_head, q_step, head_dim]
        context = context.permute(0,2,1,3).reshape((batch_size, -1, self.n_head * self.head_dim))  # [n, step, model_dim]
        o = self.o_dense(context)   # [n, step, model_dim]
        o = self.o_drop(o)
        o = self.layer_norm(residual+o)
        return o

    def split_heads(self, x):
        x = torch.reshape(x,(x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0,2,1,3)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q,k.permute(0,1,3,2)) / (torch.sqrt(dk) + 1e-8)    # [n, n_head, step, step]
        if mask is not None:
            # change the value at masked position to negative infinity,
            # so the attention score at these positions after softmax will close to 0.
            score = score.masked_fill_(mask,-np.inf)

        self.attention = softmax(score,dim=-1)
        context = torch.matmul(self.attention,v)    # [n, num_head, step, head_dim]
        return context  # [n, step, n_head * head_dim]

class PositionWiseFFN(nn.Module):
    def __init__(self,model_dim, dropout = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)

        if style == "mlp_ffn":
            dff = model_dim*4
            self.l = nn.Linear(model_dim,dff)
            self.o = nn.Linear(dff,model_dim)
            self.dropout = nn.Dropout(dropout)
        elif style == "conv1d_ffn":
            self.c1 = nn.Conv1d(in_channels=model_dim, out_channels=model_dim, kernel_size=k_size, padding=(k_size-1)//2)

    def forward(self,x):
        if style == "mlp_ffn":
            o = relu(self.l(x))
            o = self.o(o)
            o = self.dropout(o)
            o = self.layer_norm(x + o)
        elif style == "conv1d_ffn":
            o = self.c1(x.permute(0,2,1)).permute(0,2,1)
            o = self.layer_norm(o)

        return o    # [n, step, model_dim]

class EncoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = MultiHead(n_head, emb_dim, drop_rate)
        self.ffn = PositionWiseFFN(emb_dim,drop_rate)
    
    def forward(self, xz, training, mask):
        context = self.mh(xz, xz, xz, mask, training)  # [n, step, emb_dim]
        o = self.ffn(context)
        return o

class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )    
    def forward(self, xz, training, mask):
        for encoder in self.encoder_layers:
            xz = encoder(xz,training,mask)
        return xz       # [n, step, emb_dim]

class DecoderLayer(nn.Module):
    def __init__(self,n_head,model_dim,drop_rate):
        super().__init__()
        self.mh = nn.ModuleList([MultiHead(n_head, model_dim, drop_rate) for _ in range(2)])
        self.ffn = PositionWiseFFN(model_dim,drop_rate)
    
    def forward(self,yz, xz, training, yz_look_ahead_mask,xz_pad_mask):
        dec_output = self.mh[0](yz, yz, yz, yz_look_ahead_mask, training)   # [n, step, model_dim]
        
        dec_output = self.mh[1](dec_output, xz, xz, xz_pad_mask, training)  # [n, step, model_dim]

        dec_output = self.ffn(dec_output)   # [n, step, model_dim]

        return dec_output
    
class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()

        self.num_layers = n_layer

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]
        )
    
    def forward(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        for decoder in self.decoder_layers:
            yz = decoder(yz, xz, training, yz_look_ahead_mask, xz_pad_mask)
        return yz   # [n, step, model_dim]

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        pos = np.expand_dims(np.arange(max_len),1)  # [max_len, 1]
        pe = pos / np.power(1000, 2*np.expand_dims(np.arange(emb_dim)//2,0)/emb_dim)  # [max_len, emb_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe,0) # [1, max_len, emb_dim]
        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(n_vocab,emb_dim)
        self.embeddings.weight.data.normal_(0,0.1)
        
    def forward(self, x):
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)    
        x_embed = self.embeddings(x) + self.pe  # [n, step, emb_dim]
        return x_embed  # [n, step, emb_dim]
