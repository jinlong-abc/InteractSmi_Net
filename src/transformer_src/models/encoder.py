from ..layers.encoder_layer import EncoderLayer
from ..modules.norm_residual import LayerNorm
from torch import nn



class Encoder (nn.Module):
    def __init__ (self, n_layers, n_heads, model_depth, ffn_depth, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_heads, model_depth, ffn_depth, dropout) for i in range(n_layers)])
        self.lnorm = LayerNorm(model_depth)

    def forward (self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.lnorm(x)