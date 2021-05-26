import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size:int, emb_dim:int):
        super().__init__()
        self.emb_dim = emb_dim 
        self.embed = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
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

        pe = pe.unsqueeze(0)

        # why use register_buffer
        # ans link: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/11
        # An example where I find this distinction difficult is in 
        # the context of fixed positional encodings in the Transformer 
        # model. Typically I see implementations where the fixed positional 
        # encodings are registered as buffers but I’d consider these tensors 
        # as non-learnable parameters (that should show up in the list of 
        # model parameters), especially when comparing between methods 
        # that don’t rely on such injection of fixed tensors.

        # So in general:
        # buffers = ‘fixed tensors / non-learnable parameters / stuff that does not require gradient’
        # parameters = ‘learnable parameters, requires gradient’

        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embedding relatively larger by scaling the values
        # WHY?
        #   The reason we increase the embedding values before 
        #   addition is to make the positional encoding relatively 
        #   smaller. This means the original meaning in the embedding 
        #   vector won’t be lost when we add them together
        
        x = x*math.sqrt(self.emb_dim)  

        seq_len = x.size(1)

        # add constant positional embedding to the word embedding
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)