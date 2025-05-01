from dataset import *
import torch.nn as nn
import torch.nn.functional as F
from ManualLayerNorm import ManualLayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, input_embedding_size=64) -> None:
        super().__init__()
        
        self.ln = ManualLayerNorm(input_embedding_size)


    def forward(self, x):
        



        return x