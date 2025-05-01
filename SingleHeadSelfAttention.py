import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embedding_size=64, head_dim=64):
        super().__init__()

        self.embedding_size = embedding_size
        self.head_dim = head_dim

        self.W_q = nn.Linear(embedding_size, head_dim, bias=False)
        self.W_k = nn.Linear(embedding_size, head_dim, bias=False)
        self.W_v = nn.Linear(embedding_size, head_dim, bias=False)

    def forward(self, x):
        print(f"x.shape:{x.shape}")
        Q = self.W_q(x)  # [batch, seq_len, head_dim]
        K = self.W_k(x)
        V = self.W_v(x)

        print("Q = self.W_q(x)")
        
        print(f"W_q.weight.shape:{self.W_q.weight.shape}")
        print (f"W_q:{self.W_q.weight.detach().cpu().numpy()}")
        
        print(f"x.shape:{x.shape}")
        print (f"x:{x}")

        print(f"Q.shape:{Q.shape}")
        print (f"Q:{Q}")

        # Transpose K: [batch, head_dim, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]

        scale = self.head_dim ** 0.5
        scores = scores / scale

        attn_weights = torch.softmax(scores, dim=-1)

        # Step 4: Weighted sum of V
        output = torch.matmul(attn_weights, V)  # [batch, seq_len, head_dim]

        #print("Attention scores:\n", scores.squeeze(0))
    
        return output,attn_weights.detach().numpy()
    

if __name__ == '__main__':
    #Let's say I have 3 "Patches", dimensionality 2
    e1 = torch.tensor([0.1, 0.2])
    e2 = torch.tensor([0.3, 0.4])
    e3 = torch.tensor([0.5, 0.6])

    x = torch.stack([e1, e2, e3])  # Shape: [3, 2]
    x = x.unsqueeze(0)  # Add batch dim: [1, 3, 2]

    attn = SingleHeadSelfAttention(2, 2)

    output, attn_weights = attn(x)

    print("Attention weights:\n", attn_weights.squeeze(0))

    print("Output shape:", output.shape)  # should be [1, 4, 4]
    print("output:\n", output.squeeze(0))
    
    # Plot attention weights
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_weights.squeeze(0), annot=True, cmap="Blues", xticklabels=["e1", "e2", "e3"], yticklabels=["q1", "q2", "q3"])
    plt.title("Single-Head Attention Weights")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.show()




