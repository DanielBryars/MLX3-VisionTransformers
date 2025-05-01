import torch
import torch.nn as nn

class ManualLayerNorm(nn.Module):
    def __init__(self, embedding_size, Ɛ=1e-5):
        super().__init__()

        #Initialise to 1 and 0 respectively so it is just plain normalisation
        self.gamma = nn.Parameter(torch.ones(embedding_size))
        self.beta = nn.Parameter(torch.zeros(embedding_size))
        self.Ɛ = Ɛ

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.Ɛ)
        return self.gamma * x_norm + self.beta
    

if __name__ == '__main__':
    
    embedding_size = 3
    x = torch.tensor([[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]]]) 

    print(f"x.mean: {x.mean(dim=-1, keepdim=True)}")

    ln = ManualLayerNorm(embedding_size=embedding_size)

    print(f"gamma:{ln.gamma} beta:{ln.beta}")

    '''
Mean = 2.0
Variance = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1 + 0 + 1) / 3 = 0.6667
Std dev = √0.6667 ≈ 0.8165
Normalized: [(1-2)/0.8165, (2-2)/0.8165, (3-2)/0.8165] ≈ [-1.2247, 0.0, 1.2247]
    '''

    output = ln(x)

    assert output.shape == x.shape

    mean = output.mean(dim=-1)
    var = output.var(dim=-1, unbiased=False)

    print (f"output:{output}")
    print("Mean (per token):", mean)
    print("Var (per token):", var)

    #[1,16,128]