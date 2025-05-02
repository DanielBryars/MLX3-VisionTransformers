from dataset import *
import torch.nn as nn
from weights import * 

class ClassifierHead(nn.Module):

    def __init__(self,embedding_size, num_classes=10) -> None:
        super().__init__()

        #The paper says:
        #The classification head is implemented by a MLP with one hidden layer at pre-training
        #time and by a single linear layer at fine-tuning time.
        #we just keep the hidden layer in here for now.
        self.pipeline = nn.Sequential(
            nn.LayerNorm(embedding_size),

            #Hidden Layer
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            #Hidden Layer

            nn.Linear(embedding_size, num_classes))
        
    def forward(self, x):
        return self.pipeline(x)