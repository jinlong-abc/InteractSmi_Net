from ..modules.norm_residual import LayerNorm
from ..layers.decoder_layer import DecoderLayer_PerLN
from torch import nn


class Decoder (nn.Module):
    def __init__ (self, n_layers, n_heads, model_depth, ff_depth, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer_PerLN(n_heads, model_depth, ff_depth, dropout) for i in range(n_layers)])
        self.lnorm = LayerNorm(model_depth)

    def forward (self, x, src_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, src_out, src_mask, tgt_mask)
        return self.lnorm(x)