import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatMLP(nn.Module):
    """直接拼接 cf, pf → MLP"""
    def __init__(self, d, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2*d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cf, pf):
        x = torch.cat([cf, pf], dim=1)
        return self.fc(x)


class HadamardMLP(nn.Module):
    """Hadamard 乘积 cf*pf → MLP"""
    def __init__(self, d, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cf, pf):
        x = cf * pf   # [B, D]
        return self.fc(x)


class DiffAbsConcatMLP(nn.Module):
    """拼接 [cf, pf, |cf-pf|, cf*pf] → MLP"""
    def __init__(self, d, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4*d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cf, pf):
        x = torch.cat([cf, pf, torch.abs(cf - pf), cf * pf], dim=1)
        return self.fc(x)


class BilinearOuterProduct(nn.Module):
    """外积 cf ⊗ pf → 展平 → MLP"""
    def __init__(self, d, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d*d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cf, pf):
        b = cf.size(0)
        outer = torch.matmul(cf.view(b, -1, 1), pf.view(b, 1, -1))
        x = outer.view(b, -1)
        return self.fc(x)

class OuterProductLinear(nn.Module):
    """外积 cf ⊗ pf → 展平 → LeakyReLU → Linear"""
    def __init__(self, d, negative_slope=0.1):
        super().__init__()
        self.output = nn.Linear(d * d, 1)
        self.negative_slope = negative_slope

    def forward(self, cf, pf):
        b = cf.size(0)
        outer = torch.matmul(cf.view(b, -1, 1), pf.view(b, 1, -1))
        x = outer.view(b, -1)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        return self.output(x)

class BilinearProjection(nn.Module):
    """双线性投影 cf^T W pf"""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d))
        self.fc_out = nn.Linear(1, 1)
    def forward(self, cf, pf):
        score = torch.sum(cf @ self.W * pf, dim=1, keepdim=True)
        return self.fc_out(score)


class GatedFusion(nn.Module):
    """门控融合 (cf, pf → gate → 融合)"""
    def __init__(self, d, hidden_dim=256):
        super().__init__()
        self.gate = nn.Linear(2*d, d)
        self.fc = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cf, pf):
        z = torch.cat([cf, pf], dim=1)
        g = torch.sigmoid(self.gate(z))
        fused = g * cf + (1 - g) * pf
        return self.fc(fused)


class EnsembleFusion(nn.Module):
    """集成多种方法 (concat + hadamard + diff)"""
    def __init__(self, d, hidden_dim=256):
        super().__init__()
        input_dim = 2*d + d + d
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cf, pf):
        x = torch.cat([cf, pf, cf*pf, torch.abs(cf-pf)], dim=1)
        return self.fc(x)
