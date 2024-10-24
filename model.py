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

def precompute_theta_pos_frequencies(head_dim:int,seq_len=int,device=str,theta:float=10000.0):
    assert head_dim%2==0, "Head Dimension must be even" # According to the paper the head dimension cant be applied to the odd dimension of embeddings
    # Bulding the theta parameters
    # According to the formula theta_i=100000^(-2(i-1/dim) for i= {1,2,3,...,dim/2})
    # Shape: (Head_Dim/2)
    theta_numerator=torch.arrange(0,head_dim,2).float()
    # Shape is (Head_Dim/2)
    theta=1.0/(theta**theta_numerator/head_dim).to(device)
    # Constructing the positions which is the "m" parameter
    # shape is (Seq_Len)
    m=torch.arrange(seq_len,device)
    # Muliplying each theta by each position using the outer product
    # Shape is (Seq_Len,Head_Dim/2)
    freqs=torch.outer(m,theta).float()
    # Computing Complex Numbers in the polar form c=R*exp(i*m*theta), where R=1
    freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex


def apply_rotary_embeddings(x:torch.Tensor,freqs_complex:torch.Tensor):
    # x: (Batch,Seq_Len,Head_Dim) -> (Batch,Seq_Len,Head_Dim/2,)
    # Converting the input to the complex form
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    # (Batch,Seq_Len,Head_Dim/2)->(1,1,Seq_Len,Head_Dim/2)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated=x_complex*freqs_complex
    x_out=torch.view_as_real(x_rotated)
    x_out=x_out.reshape(*x.shape)
    return x_out


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