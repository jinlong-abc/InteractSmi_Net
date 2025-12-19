from ..modules.norm_residual import ResidualNorm
from ..modules.attention import MultiHeadAttention
from ..modules.feedforward import FFN
from torch import nn




class EncoderLayer (nn.Module):
    def __init__ (self, n_heads, model_depth, ffn_depth, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm1 = ResidualNorm(model_depth, dropout)
        self.ffn = FFN(model_depth, ffn_depth, dropout)
        self.resnorm2 = ResidualNorm(model_depth, dropout)

    def forward (self, x, mask):
        x = self.resnorm1(x, lambda arg: self.self_attn(arg, arg, arg, mask))
        x = self.resnorm2(x, self.ff)
        return x