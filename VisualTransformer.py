from PatchEmbedder import *
from TransformerBlock import *
from ClassifierHead import *

class VisualTransformer(nn.Module):
    def __init__(self,patch_size, embedding_size, num_classes) -> None:
        super().__init__()

        self.patch_embedder = PatchEmbedder(patch_size, embedding_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.transformer1 = TransformerBlock(embedding_size)
        self.transformer2 = TransformerBlock(embedding_size)
        self.classifier_head = ClassifierHead(embedding_size, num_classes)
        
    def forward(self, x):
        x = self.patch_embedder(x)   # x: [batch_size, num_patches, embedding_size]

        batch_size = x.shape[0]

        # Expand cls_token across batch dimension
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, embedding_size]

        # Prepend CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # x: [batch_size, num_patches+1, embedding_size]

        # Forward through Transformer(s)
        x = self.transformer1(x)
        x = self.transformer2(x)

        # Take only the CLS token
        cls_output = x[:, 0, :]  # shape: [batch_size, embedding_size]

        # Classify
        logits = self.classifier_head(cls_output)

        return logits

