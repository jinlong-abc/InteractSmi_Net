import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_depth, dropout=0.1):
        super().__init__()
        assert model_depth % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_depth = model_depth
        self.num_heads = num_heads
        self.head_dim = model_depth // num_heads
        
        # 初始化投影(projection)矩阵
        self.q_proj = nn.Linear(model_depth, model_depth)
        self.k_proj = nn.Linear(model_depth, model_depth)
        self.v_proj = nn.Linear(model_depth, model_depth)
        self.out_proj = nn.Linear(model_depth, model_depth)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, mask=None):
        '''
        mask=1表示可见，mask=0表示屏蔽
        '''
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size = query.size(0)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
        Q = self._split_heads(self.q_proj(query), batch_size) 
        K = self._split_heads(self.k_proj(key), batch_size)
        V = self._split_heads(self.v_proj(value), batch_size)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        context = self._combine_heads(context, batch_size)
        return self.out_proj(context), attn_weights

    def _split_heads(self, x, batch_size):
        """将投影后的向量分割为多个头, view方法需要明确指定所有维度大小"""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x, batch_size):
        """合并多个头"""
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.model_depth)


