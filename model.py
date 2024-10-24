import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim:int=4096
    n_layers:int=32
    n_heads:int=32   # Number of Heads for Queries
    n_kv_heads:Optional[int]=None # Number of Heads for Keys and Values
    vocab_size:int=-1 # This will be set when we load the tokenizer
    multiple_of:int=226 
    ffn_dim_multiplier:Optional[int]=None    # This and the line above indicates the hidden dimension of the feed forward neural network
    norm_eps:float=1e-5

    # Required for KV Cache
    max_batch_size:int=32
    max_seq_len:int=2048
    device:str=None

# Complete Model Except the Softmax
class Transformer(nn.Module):
    def __init__(self,args:ModelArgs)->None:
        super().__init__()

        assert args.vocab_size!=-1, "Vocab Size must be set"

        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        self.tok_embeddings=nn.Embedding(self.vocab_size,args.dim)

        self.layers=nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.normalization=RMSNorm(args.dim,eps=args.norm_eps)
        self.output=nn.Linear(args.dim,self.vocab_size,bias=False)

        self.freqs_complex=precompute_theta_pos_frequencies(self.args.dim//self.args.n_heads,self.args.max_seq_len*2,device=self.args.device)  # For Positional Encoding

    def forward(self,tokens:torch.Tensor,start_pos:int):
        # (Batch,Sequence Length)
        batch_size,seq_len=tokens.shape
        assert seq_len==1, "Sequence Length must be 1/Only 1 token at a time can be processed" 

        # (Batch,Sequence Length,Embedding Dimension) Convert Tokens to Embeddings
        h=self.tok_embeddings(tokens)

        # Retrive the pairs (m,theta) coresponding to the position [Start_pos,start_pos+seq_len]
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]

        # Consecutively applying to all the encoder layers
        for layer in self.layers:
            h=layer(h,start_pos,freqs_complex)
        h=self.norm(h)
        output=self.output(h).float()
        return output