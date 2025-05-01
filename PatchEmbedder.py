from dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from weights import * 

class PatchEmbedderDan(nn.Module):
    def __init__(self,patch_size, image_size, embedding_size) -> None:
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.image_size = image_size
        self.linear = nn.Linear(patch_size**2, embedding_size)
        assert image_size % patch_size == 0, "patch size must be a factor of image size"
        self.num_patches = (image_size // patch_size) ** 2

        #create some position vectors
        
    def to_unit_range(self, x):
        return x.float() / 255.0

    def normalise(self, x):
        mnist_std = 0.3081
        mnist_mean = 0.1307
        x = (x - mnist_mean) / mnist_std
        return x

    def forward(self, x):
        assert x.shape[1] == 1, "This implementation assumes single channel greyscale image 0-255"
        assert x.shape[2] == self.image_size and x.shape[3] == self.image_size

        #convert to floats and rescale to 0-1
        x = self.to_unit_range(x)

        #normalise so they are not so dark
        x = self.normalise(x)

        #Patching        
        x = self.unfold(x)

        #unfolding gives patches in last dimension
        x = x.transpose(1, 2)
        
        #Forward through the Trainable Linear Layer
        x = self.linear(x)

        return x

class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, image_size, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_dim = embedding_dim

        self.proj = nn.Conv2d(
            in_channels=1,  # MNIST is grayscale
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        return x

if __name__ == '__main__':

    patch_size = 14
    image_size = 28
    embedding_size= 196
    pe = PatchEmbedder(patch_size, image_size, embedding_size)

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

    #plot_weight_matrix(pe.linear)

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