from models.PatchEmbedder import *
from models.TransformerBlock import *
from models.ClassifierHead import *
from models.PosistionEncoder import *

class VisualTransformer(nn.Module):
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

        self.attn_weights = []
        image_size = 28
        self.patch_embedder = PatchEmbedder(patch_size, image_size, embedding_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size)**2 + 1, embedding_size))
        self.dropout = nn.Dropout(dropout)

        print(f"Initialising VisualTransformer with num_transformer_blocks:{num_transformer_blocks}")
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_size, num_heads, mlp_dim, dropout) for _ in range(num_transformer_blocks)]
        )

        self.norm = nn.LayerNorm(embedding_size)
        self.classifier_head = ClassifierHead(embedding_size, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attn=False):
        x = self.patch_embedder(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.dropout(x)
        #x = self.transformer_blocks(x)
        i =0
        for block in self.transformer_blocks:
            if return_attn:
                x, attn = block(x, return_attention=True)
                self.attn_weights.append(attn)
                i+=1
                print(f"Forward through block:{i}")
            else:
                x = block(x)

        x = self.norm(x)
        return self.classifier_head(x[:, 0])

class DjbVisualTransformer(nn.Module):
    def __init__(
            self,
            patch_size, 
            embedding_size,
            num_classes, 
            txBlock,
            num_transformer_blocks, 
            num_heads, 
            mlp_dim, 
            dropout) -> None:
        super().__init__()

        image_size = 28
        self.patch_embedder = PatchEmbedder(patch_size, image_size, embedding_size)

        self.position_encoder = PosistionEncoder(patch_size, image_size, embedding_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

        self.transformer_blocks = nn.Sequential(*[txBlock(embedding_size,num_heads, mlp_dim, dropout) for _ in range(num_transformer_blocks)])
        
        self.classifier_head = ClassifierHead(embedding_size, num_classes)
        
    def forward(self, x, return_attention=False):
        x = self.patch_embedder(x)   # x: [batch_size, num_patches, embedding_size]

        x = self.position_encoder(x)

        batch_size = x.shape[0]

        # Expand cls_token across batch dimension
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embedding_size]

        # Prepend CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # x: [batch_size, num_patches+1, embedding_size]

        # Forward through Transformer(s)


        x = self.transformer_blocks(x)
        
        # Take only the CLS token
        cls_output = x[:, 0, :]  # shape: [batch_size, embedding_size]

        # Classify
        logits = self.classifier_head(cls_output)

        return logits

