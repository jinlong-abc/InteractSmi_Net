from ..modules.norm_residual import ResidualNorm_PostLN, ResidualNorm_PreLN
from ..modules.attention import MultiHeadAttention
from ..modules.feedforward import FFN
from torch import nn




class DecoderLayer_PostLN (nn.Module):
    '''x + Norm(Sublayer(x))'''
    def __init__ (self, n_heads, model_depth, ff_depth, dropout):
        super(DecoderLayer_PostLN, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm1 = ResidualNorm_PostLN(model_depth, dropout)
        self.enc_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm2 = ResidualNorm_PostLN(model_depth, dropout)
        self.ff = FFN(model_depth, ff_depth, dropout)
        self.resnorm3 = ResidualNorm_PostLN(model_depth, dropout)

    def forward (self, x, src_out, src_mask, tgt_mask): 
        x = self.resnorm1(x, lambda arg: self.self_attn(arg, arg, arg, tgt_mask)[0])
        x = self.resnorm2(x, lambda arg: self.enc_attn(arg, src_out, src_out, src_mask)[0])
        x = self.resnorm3(x, self.ff)
        return x
    


class DecoderLayer_PerLN (nn.Module):
    '''x + Sublayer(Norm(x))'''
    def __init__ (self, n_heads, model_depth, ff_depth, dropout):
        super(DecoderLayer_PerLN, self).__init__()
        self.self_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm1 = ResidualNorm_PreLN(model_depth, dropout)
        self.enc_attn = MultiHeadAttention(n_heads, model_depth)
        self.resnorm2 = ResidualNorm_PreLN(model_depth, dropout)
        self.ff = FFN(model_depth, ff_depth, dropout)
        self.resnorm3 = ResidualNorm_PreLN(model_depth, dropout)

    def forward(self, x, src_out, src_mask, tgt_mask):
        x = x + self.resnorm1.dropout(
            self.self_attn(self.resnorm1.norm(x), 
                          self.resnorm1.norm(x),
                          self.resnorm1.norm(x), tgt_mask)[0]
        )

        x = x + self.resnorm2.dropout(
            self.enc_attn(self.resnorm2.norm(x),
                         self.resnorm2.norm(src_out),
                         self.resnorm2.norm(src_out), src_mask)[0]
        )

        x = x + self.resnorm3.dropout(
            self.ff(self.resnorm3.norm(x))
        )
        return x