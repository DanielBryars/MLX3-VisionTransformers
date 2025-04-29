from dataset import *
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, input_embedding_size=64) -> None:
        super().__init__()
        



    def forward(self, x):
        print(f"x.shape:{x.shape}")
        
        


        return x