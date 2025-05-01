from ClassifierHead import nn, torch
from PatchEmbedder import math, nn, torch
from TransformerBlock import nn


class PosistionEncoder(nn.Module):
    def __init__(self, patch_size, image_size, embedding_size):

        position_vectors = torch.empty(self.num_patches, embedding_size)

        for patch_idx in range(self.num_patches):
            position_vector = position_vectors[patch_idx]
            for embedding_component_idx in range(embedding_size):
                angle = patch_idx / 10000 ** (embedding_component_idx/(embedding_size//2))
                if embedding_component_idx % 2 == 0:
                    #even
                    position_vector[embedding_component_idx] = math.sin(angle)
                else:
                    position_vector[embedding_component_idx] = math.cos(angle)

        #Can then reference this using self.position_vectors      
        self.register_buffer("position_vectors", position_vectors)

    def forward(self, x):

        #add a position vector onto each embedding
        #do it in a dumb way
        #self.position_vectors

        x += self.position_vectors

        return x