from ..modules.generator import Generator
from ..modules.embeddings import EmbeddingWithPositionalEncoding
from ..models.decoder import Decoder
from ..models.encoder import Encoder
from torch import nn


class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, model_depth, ff_depth, vocab_size, dropout):
        super(Transformer, self).__init__()
        self.embedwithpos = EmbeddingWithPositionalEncoding(model_depth, vocab_size)
        self.encoder = Encoder(n_layers, n_heads, model_depth, ff_depth, dropout)
        self.decoder = Decoder(n_layers, n_heads, model_depth, ff_depth, dropout)
        self.generator = Generator(model_depth, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.embedwithpos(src)
        tgt = self.embedwithpos(tgt)
        src_out = self.encoder(src, src_mask)
        tgt_out = self.decoder(tgt, src_out, src_mask, tgt_mask)
        return self.generator(tgt_out)