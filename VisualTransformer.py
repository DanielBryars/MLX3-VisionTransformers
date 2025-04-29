from PatchEmbedder import *
from TransformerBlock import *
from ClassifierHead import *

class VisualTransformer():
    def __init__(self,patch_size, embedding_size, num_classes) -> None:
        super().__init__()

        self.pipeline = nn.Sequential(
            PatchEmbedder(patch_size, embedding_size),
            TransformerBlock(embedding_size),
            TransformerBlock(embedding_size),
            ClassifierHead(embedding_size,num_classes))
        
    def forward(self, x):
        return self.pipeline(x)


'''
This is what I 'm going to build

patch_embedder = PatchEmbedder(...)
transformer_block1 = TransformerBlock(...)
transformer_block2 = TransformerBlock(...)
...
classifier = ClassifierHead(...)

x image [28,28]
x = patch_embedder(x) #patch + linear layer
x [nP*64] where nP is the number of patches
x = transformer_block1(x)

x [nP*64]
x = transformer_block2(x)

x [nP*64]

x = classifier(x)
x [nC] where nC is the numbe of classes (labels)

'''

