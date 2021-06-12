import numpy as np

import torch
import torch.nn as nn
import math
from torch.autograd import Variable

import torch.nn.functional as F
import copy


class Embedder(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int):
        super().__init__()
        self.emb_dim = emb_dim 
        self.embed = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x: torch.tensor):
        return self.embed(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim:int, max_seq_len:int = 200, dropout_pct:float = 0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout_pct)

        # create constant 'pe' matrix with values dependent on 
        # word position 'pos' and embedding position 'i'
        pe = torch.zeros(max_seq_len, emb_dim)

        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                pe[pos,i] = math.sin(pos/(1000**((2*i)/emb_dim)))
                pe[pos,i+1] = math.sin(pos/(1000**((2*(i+1))/emb_dim)))

        # print(pe.size())
        # pe = pe.unsqueeze(0)
        # print(pe.size())

        self.register_buffer('pe', pe)


    def forward(self, x):
        
        # scale values
        x = x*math.sqrt(self.emb_dim)  

        seq_len = x.size(0)
        
        # add constant positional embedding to the word embedding
        pe = Variable(self.pe[:seq_len,:], requires_grad=False)
        
        if x.is_cuda:
            pe.cuda()
        
        x = x + pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        
        x_mean = x.mean(dim=-1, keepdim=True)
        x_variance = x.std(dim=-1, keepdim=True) 
        
        normalized_x = (x - x_mean) / (x_variance + self.eps)
        
        # scale and shift
        y = self.alpha * normalized_x + self.bias
        return y

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int = 2048, dropout_pct:float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_pct)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout_pct = 0.1):
        super().__init__()
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout_pct)
        
    def forward(self, x, attn_output):
        # add
        x = x + self.dropout(attn_output)
        
        # normalize
        x = self.norm(x)
        
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout_pct = 0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout_pct)
        self.ff = FeedForward(d_model, dropout_pct = dropout_pct)
        self.add_norm_1 = AddNorm(d_model, dropout_pct)
        self.add_norm_2 = AddNorm(d_model, dropout_pct)
        
    def forward(self, x, mask):
        # q = k = v = x
        x = self.add_norm_1(x,self.attn(x,x,x,mask))
        x = self.add_norm_2(x, self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout_pct = 0.1):
        super().__init__()
        
        self.attn_decoder = MultiHeadAttention(heads, d_model)
        self.attn_encoder = MultiHeadAttention(heads, d_model)
        
        self.ff = FeedForward(d_model)
        
        self.add_norm_1 = AddNorm(d_model, dropout_pct)
        self.add_norm_2 = AddNorm(d_model, dropout_pct)
        self.add_norm_3 = AddNorm(d_model, dropout_pct)
        
        
        
    def forward(self, x, encoder_output, src_mask, trg_mask):
        
        """
        x: this x comes from the target language
        """
        
        q = k = v = x
        x = self.add_norm_1(x ,self.attn_decoder(q,k,v,trg_mask))
        
        k_enc = v_enc = encoder_output
        x = self.add_norm_2(x, self.attn_encoder(x, k_enc, v_enc, src_mask))
        
        x = self.add_norm_3(x, self.ff(x))
        
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_encoder_layer, heads, dropout):
        super().__init__()
        self.n_encoder_layer = n_encoder_layer
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEmbedding(d_model)
        enc_layer = EncoderLayer(d_model, heads, dropout)
        # stack enc_layers
        self.enc_layers = get_clones(enc_layer, n_encoder_layer)
        
    def forward(self, src_token, mask):
        x = self.embed(src_token)
        x = self.pe(x)
        
        for i in range(self.n_encoder_layer):
            x = self.enc_layers[i](x, mask)
        
        # the EncoderLayer return normalized x
        # so no need to pass through Norm() again
        
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_decoder_layer, heads, dropout):
        super().__init__()
        self.n_decoder_layer = n_decoder_layer
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEmbedding(d_model)
        dec_layer = DecoderLayer(d_model, heads, dropout_pct=dropout)
        self.dec_layers = get_clones(dec_layer, n_decoder_layer)
        self.norm = Norm(d_model)
        
    def forward(self,trg, enc_output, src_mask, trg_mask):
        x = self.pe(self.embed(trg))
        for i in range(self.n_decoder_layer):
            x = self.dec_layers[i](x, enc_output, src_mask, trg_mask)
          
        # the DecoderLayer returns normalized x
        # so no need to pass through Norm() again
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, n_layers, heads, dropout = 0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, n_layers, heads, dropout)
        self.linear = nn.Linear(d_model, trg_vocab)
        
    def forward(self, src, trg, src_mask, trg_mask):
        
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.linear(dec_output)
        
        return output
    
# we don't perform softmax on the output as this will be handled 
# automatically by our loss function

