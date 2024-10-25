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
def repeat_kv(self,x:torch.Tensor,n_rep:int)->torch.Tensor:
    batch_size,seq_len,n_kv_heads,head_dim=x.shape
    if n_rep==1:
        return x
    else:
        return (
            x[:,:,:,None,:]
            .expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim)
            .reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
        )
    
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    def norm(self,x:torch.Tensor):
        # rsqrt:1/sqrt(x)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x:torch.Tensor):
        return self._norm(x.float()).type_as(x)*self.weight

class SelfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()

        # Indicates the number of heads for keys and values
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for queries
        self.n_heads_q=args.n_heads
        # Indicates how many times the head of keys and values should be repeated to match the head of queries
        self.n_rep=self.n_heads_q//self.n_kv_heads
        # Indicate the dimensions of each head
        self.head_dim=args.dim/args.n_heads

        self.wq=nn.linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.linear(args.dim,args.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.linear(args.dim,args.n_kv_heads*self.head_dim,bias=False)
        self.wo=nn.linear(args.n_heads*args.head_dim,args.dim,bias=False)

        self.cache_k=torch.zeros((args.max_batch_size,args.max_seq_len,args.n_kv_heads,args.head_dim))
        self.cache_v=torch.zeros((args.max_batch_size,args.max_seq_len,args.n_kv_heads,args.head_dim))

    
    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
        batch_size,seq_len,_=x.shape # (Batch,1,Dim)
        # Applying the linear transformation to the queries, keys and values
        # (B,1,dim)->(B,1,h_q*head_dim)
        xq=self.wq(x)           
        # (B,1,dim)->(B,1,h_kv*head_dim)
        xk=self.wk(x)
        # (B,1,dim)->(B,1,h_kv*head_dim)
        xv=self.wv(x)
        xq=xq.view(batch_size,seq_len,self.n_heads_q,self.head_dim)
        xk=xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim) 
        xq=apply_rotary_embeddings(xq,freqs_complex,device=x.device)
        xk=apply_rotary_embeddings(xk,freqs_complex,device=x.device)

        self.cache_k[:batch_size,:start_pos:start_pos+seq_len]=xk
        self.cache_v[:batch_size,:start_pos:start_pos+seq_len]=xv

        # Retrieving the keys and values from the cache
        keys=self.cache_k[:batch_size,0:start_pos+seq_len]
        values=self.cache_v[:batch_size,0:start_pos+seq_len]

        # Repeat the heads of the K and V to reach the number of head in queries
        keys=repeat_kv(keys,self.n_rep)
        values=repeat_kv(values,self.n_rep)

        xq=xq.transpose(1,2)
        keys=keys.transpose(1,2)
        values=values.transpose(1,2)

        scores=torch.matmul(xq,keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)
        output=torch.matmul(scores,values)
        output=(output.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(output)
    
class FeedForward(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        hidden_dim=4*args.dim
        hidden_dim=int(2*hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim=int(args.ffn_dim_multiplier*args.dim)
        # Round the hidden_dim to the nearest multiple of the parameter
        hidden_dim=args.multiple_of*((hidden_dim+args.multiple_of-1)//args.multiple_of)
        self.w1=nn.linear(args.dim,hidden_dim,bias=False)
        self.w2=nn.linear(hidden_dim,args.dim,bias=False)
        self.w3=nn.linear(args.dim,args.dim,bias=False)

    def forward(self,x:torch.Tensor):
        swish=F.silu(self.w1(x))
        x_V=self.w3(x)
        x=swish*x_V
        x=self.w2(x)
        return x
   
class EncoderBlock(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads

        self.attention=SelfAttention(args)
        self.feed_forward=FeedForward(args)

        # Normalization before self attention
        self.attention_norm=RMSNorm(args.dim,eps=args.norm_eps)

        # Normalization before feed forward
        self.ffn_norm=RMSNorm(args.dim,eps=args.norm_eps)
    
    def forward(self,x:torch.Tensor,start_pos:int,freqs_complex:torch.Tensor):
      h=x+self.attention.feed_forward(self.attention_norm(x),start_pos,freqs_complex)
      out=h+self.feed_forward.forward(self.ffn_norm(h))
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