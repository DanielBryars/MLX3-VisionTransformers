from dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from weights import * 

class PatchEmbedder(nn.Module):
    def __init__(self,patch_size, embedding_size=64) -> None:
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size**2, embedding_size)

    def forward(self, x):
        assert x.shape[1] == 1, "This implementation assumes single channel greyscale image 0-255"

        #convert to floats and rescale to 0-1
        x = x.float() / 255.0

        #normalise so they are not so dark
        mnist_std = 0.3081
        mnist_mean = 0.1307
        x = (x - mnist_mean) / mnist_std

        #Patching        
        x = self.unfold(x)

        x = x.transpose(1, 2)
        
        #Linear Layer
        x = self.linear(x)

        return x

if __name__ == '__main__':

    patch_size = 7
    embedding_size= 64
    pe = PatchEmbedder(patch_size,embedding_size)

    #for testing only setup the weights to "transparently" pass the image data through
    init_projection_weights(pe.linear)

    def plot_weight_matrix(linear):
        weights = linear.weight.detach().cpu().numpy()  # shape: (out_dim, in_dim)
        plt.figure(figsize=(6, 6))
        plt.imshow(weights, aspect='auto', cmap='hot')
        plt.title("Weight Matrix Heatmap")
        plt.xlabel("Input Dimension")
        plt.ylabel("Output Dimension")
        plt.colorbar()
        plt.show()

    plot_weight_matrix(pe.linear)

    pe.eval()

    image, label = mnist_train[0]
    print(f"image.shape:{image.shape}")

    output = pe.forward(image.unsqueeze(0)) # unsqueeze to make it a "batch of 1"
    print(f"output.shape:{output.shape}")

    #print(output)

    #now lets visualise this.
    output = output.squeeze()
    #[16,64]

    num_patches = output.shape[0]
    num_rows = num_cols = int(math.sqrt(num_patches))  # Assuming square grid

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_cols, 2*num_rows))
    axes = axes.flatten()

    #output is an embedding, let's assume it's square and display it as if it were an image
    #embedding_size
    image_size = int(math.sqrt(embedding_size))
    for idx in range(num_patches):
        embedding = output[idx] 
        patch = embedding.reshape(image_size, image_size)
        patch = patch.detach().cpu().numpy() #bring it back
        #print (f"patch {idx}:")
        #print (patch)
        axes[idx].imshow(patch, cmap='grey')
        axes[idx].axis('off')

    plt.suptitle(f"MNIST digit: {label}")
    plt.show()