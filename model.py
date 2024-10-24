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

