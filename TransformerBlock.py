from dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class TransformerBlock(nn.Module):
    def __init__(self, input_embedding_size=64) -> None:
        super().__init__()
        
    def forward(self, x):
        return x