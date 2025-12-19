import torch
import torch.nn as nn



class LayerNorm(nn.Module):
    "对每个 token 的特征维度进行归一化"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x): 
        mean = x.mean(-1, keepdim=True) 
        var = x.var(-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return self.a_2 * (x - mean) / std + self.b_2 

class ResidualNorm_PostLN (nn.Module):
    def __init__ (self, size, dropout):
        super(ResidualNorm_PostLN, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class ResidualNorm_PreLN (nn.Module):
    def __init__ (self, size, dropout):
        super(ResidualNorm_PreLN, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))