import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)

class BiAttentionPooling(nn.Module):
    def __init__(self, dim1, dim2, fusion='concat', hidden_dim=256):
        super().__init__()
        self.pool1 = AttentionPooling(dim1)
        self.pool2 = AttentionPooling(dim2)
        self.fusion = fusion

        if fusion == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(dim1 + dim2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(self, x1, x2):
        v1 = self.pool1(x1)
        v2 = self.pool2(x2)

        if self.fusion == 'concat':
            out = torch.cat([v1, v2], dim=-1)  # [B, D1 + D2]
        elif self.fusion == 'sum':
            assert v1.size(1) == v2.size(1), "sum fusion requires equal dims"
            out = v1 + v2  # [B, D]
        elif self.fusion == 'mlp':
            out = torch.cat([v1, v2], dim=-1)  # [B, D1 + D2]
            out = self.mlp(out)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")

        return out  
