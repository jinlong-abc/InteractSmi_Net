# 1. 位置编码和词嵌入
# filepath: src/modules/embeddings.py
import torch
import torch.nn as nn
import math



class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, model_depth):
        super(TokenEmbedding, self).__init__()
        if not isinstance(vocab_size, int) or not isinstance(model_depth, int):
            raise ValueError(f"vocab_size and model_depth must be integers, got vocab_size={vocab_size}, model_depth={model_depth}")
        
        self.lut = nn.Embedding(vocab_size, model_depth) 
        self.model_depth = model_depth
        nn.init.xavier_uniform_(self.lut.weight)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.model_depth)
    


class PositionalEncoding(nn.Module):
    def __init__(self, model_depth, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, model_depth)
        position = torch.arange(0.0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0.0, model_depth, 2) *  
                             -(math.log(10000.0) / model_depth)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, model_depth):
        super(EmbeddingWithPositionalEncoding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, model_depth)
        self.positional_encoding = PositionalEncoding(model_depth)

    def forward(self, x):
        emb = self.token_embedding(x)
        return self.positional_encoding(emb)