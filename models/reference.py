import torch
import torch.nn as nn

class PatchEmbedder2(nn.Module):
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

import torch
import torch.nn as nn

class TransformerBlock2(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res
        return x

import torch.nn as nn

class ClassifierHead2(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    

import torch
import torch.nn as nn
from models.PatchEmbedder import PatchEmbedder
from models.TransformerBlock import TransformerBlock
from models.ClassifierHead import ClassifierHead

class VisualTransformer2(nn.Module):
    def __init__(
        self,
        patch_size=4,
        embedding_size=256,
        num_classes=10,
        num_transformer_blocks=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1
    ):
        super().__init__()

        image_size = 28
        self.patch_embedder = PatchEmbedder(patch_size, image_size, embedding_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, embedding_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock2(embedding_size, num_heads, mlp_dim, dropout) for _ in range(num_transformer_blocks)]
        )

        self.norm = nn.LayerNorm(embedding_size)
        self.classifier_head = ClassifierHead2(embedding_size, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embedder(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        return self.classifier_head(x[:, 0])

