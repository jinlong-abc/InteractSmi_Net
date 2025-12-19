import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================
# 基础注意力机制模块
# ============================

class CrossAttentionLayer(nn.Module):
    """
    标准的交叉注意力层
    用于计算两个序列之间的注意力关系
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(CrossAttentionLayer, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [batch_size, query_len, d_model]
            attention: [batch_size, num_heads, query_len, key_len]
        """
        batch_size, query_len = query.size(0), query.size(1)
        
        Q = self.W_q(query).view(batch_size, query_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
    
        context = torch.matmul(attention, V)
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        output = self.W_o(context)
        
        return output, attention


class CoAttentionLayer(nn.Module):
    """
    协同注意力层
    同时计算两个模态之间的相互注意力
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super(CoAttentionLayer, self).__init__()
        self.d_model = d_model
        
        # 特征变换层
        self.W_comp = nn.Linear(d_model, d_model)
        self.W_prot = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.W_comp, self.W_prot]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(self, comp_feat: torch.Tensor, prot_feat: torch.Tensor,
                comp_mask: Optional[torch.Tensor] = None, 
                prot_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        计算协同注意力
        """
        comp_trans = torch.tanh(self.W_comp(comp_feat))
        prot_trans = torch.tanh(self.W_prot(prot_feat))
        
        affinity = torch.matmul(comp_trans, prot_trans.transpose(1, 2))
        
        if comp_mask is not None and prot_mask is not None:
            mask_matrix = torch.matmul(
                comp_mask.unsqueeze(2).float(), 
                prot_mask.unsqueeze(1).float()
            )
            affinity = affinity.masked_fill(mask_matrix == 0, -1e9)
        
        comp_attention = F.softmax(affinity, dim=-1)
        prot_attention = F.softmax(affinity.transpose(1, 2), dim=-1)
        
        comp_attended = torch.matmul(comp_attention, prot_feat)
        prot_attended = torch.matmul(prot_attention, comp_feat)
        
        return comp_attended, prot_attended, comp_attention, prot_attention


class GatedFusionLayer(nn.Module):
    """
    门控融合层
    使用门控机制融合原始特征和交叉注意力特征
    """
    def __init__(self, d_model: int):
        super(GatedFusionLayer, self).__init__()
        self.fusion_comp = nn.Linear(d_model * 2, d_model)
        self.fusion_prot = nn.Linear(d_model * 2, d_model)
        self.gate_comp = nn.Linear(d_model * 2, d_model)
        self.gate_prot = nn.Linear(d_model * 2, d_model)
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.fusion_comp, self.fusion_prot, self.gate_comp, self.gate_prot]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(self, comp_orig: torch.Tensor, comp_cross: torch.Tensor,
                prot_orig: torch.Tensor, prot_cross: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控融合原始特征和交叉注意力特征
        """
        comp_combined = torch.cat([comp_orig, comp_cross], dim=-1)
        comp_gate = torch.sigmoid(self.gate_comp(comp_combined))
        comp_fused = comp_gate * self.fusion_comp(comp_combined) + (1 - comp_gate) * comp_orig
        
        prot_combined = torch.cat([prot_orig, prot_cross], dim=-1)
        prot_gate = torch.sigmoid(self.gate_prot(prot_combined))
        prot_fused = prot_gate * self.fusion_prot(prot_combined) + (1 - prot_gate) * prot_orig
        
        return comp_fused, prot_fused


# ============================
# 融合策略实现模块
# ============================

class BidirectionalAttentionModule(nn.Module):
    """双向注意力模块"""
    
    def __init__(self, latent_dim: int, num_layers: int = 4):
        super(BidirectionalAttentionModule, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(latent_dim, latent_dim)) 
            for _ in range(num_layers)
        ])
        
        self.transform_c2p = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(num_layers)
        ])
        self.transform_p2c = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(num_layers)
        ])
        
        self.hidden_comp = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(num_layers)
        ])
        self.hidden_prot = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(num_layers)
        ])
        
        self.attention_comp = nn.ModuleList([
            nn.Linear(latent_dim * 2, 1) for _ in range(num_layers)
        ])
        self.attention_prot = nn.ModuleList([
            nn.Linear(latent_dim * 2, 1) for _ in range(num_layers)
        ])
        
        self.combine_comp = nn.Linear(latent_dim * num_layers, latent_dim)
        self.combine_prot = nn.Linear(latent_dim * num_layers, latent_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for u_matrix in self.U:
            nn.init.xavier_uniform_(u_matrix, gain=1.414)
            
    def forward(self, comp_feat: torch.Tensor, comp_mask: torch.Tensor,
                prot_feat: torch.Tensor, prot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = comp_feat.shape[0]
        comp_features = []
        prot_features = []

        comp_layers = []
        prot_layers = []

        for i in range(self.num_layers):
            attention_matrix = torch.tanh(torch.matmul(
                torch.matmul(comp_feat, self.U[i]), 
                prot_feat.transpose(1, 2)
            ))
            mask_matrix = torch.matmul(
                comp_mask.view(batch_size, -1, 1).float(),
                prot_mask.view(batch_size, 1, -1).float()
            )
            attention_matrix = attention_matrix * mask_matrix
            
            comp_cross = torch.matmul(
                attention_matrix, 
                torch.tanh(self.transform_p2c[i](prot_feat))
            )
            prot_cross = torch.matmul(
                attention_matrix.transpose(1, 2),
                torch.tanh(self.transform_c2p[i](comp_feat))
            )
            
            comp_combined = torch.cat([
                torch.tanh(self.hidden_comp[i](comp_feat)), comp_cross
            ], dim=2)
            prot_combined = torch.cat([
                torch.tanh(self.hidden_prot[i](prot_feat)), prot_cross
            ], dim=2)
            
            comp_layers.append(comp_combined)
            prot_layers.append(prot_combined)

            comp_weights = self._masked_softmax(
                self.attention_comp[i](comp_combined).view(batch_size, -1),
                comp_mask.view(batch_size, -1)
            )
            prot_weights = self._masked_softmax(
                self.attention_prot[i](prot_combined).view(batch_size, -1),
                prot_mask.view(batch_size, -1)
            )
            
            comp_pooled = torch.sum(comp_feat * comp_weights.view(batch_size, -1, 1), dim=1)
            prot_pooled = torch.sum(prot_feat * prot_weights.view(batch_size, -1, 1), dim=1)
                       
            comp_features.append(comp_pooled) 
            prot_features.append(prot_pooled)
        
        comp_layerwise = torch.cat(comp_layers, dim=2)
        prot_layerwise = torch.cat(prot_layers, dim=2)

        # 组合所有层的特征_亲和力预测使用
        comp_final = self.combine_comp(torch.cat(comp_features, dim=1))
        prot_final = self.combine_prot(torch.cat(prot_features, dim=1))
        
        return comp_final, prot_final, comp_layerwise, prot_layerwise
    
    def _masked_softmax(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """带掩码的softmax"""
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        scores_exp = torch.exp(scores - scores_max)
        scores_exp = scores_exp * mask.float()
        return scores_exp / (torch.sum(scores_exp, dim=-1, keepdim=True) + 1e-6)

class BilinearAttentionNetwork(nn.Module):
    """双线性注意力网络(BAN)模块"""
    
    def __init__(self, comp_dim: int, prot_dim: int, hidden_dim: int, num_layers: int = 3):
        super(BilinearAttentionNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.comp_projections = nn.ModuleList([
            nn.Linear(comp_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.prot_projections = nn.ModuleList([
            nn.Linear(prot_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.bilinear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)
        ])
        
        self.comp_attention = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        self.prot_attention = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        
        self.comp_combine = nn.Linear(num_layers * hidden_dim, hidden_dim)
        self.prot_combine = nn.Linear(num_layers * hidden_dim, hidden_dim)
        
    def forward(self, comp_feat: torch.Tensor, comp_mask: torch.Tensor,
                prot_feat: torch.Tensor, prot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = comp_feat.shape[0]
        comp_features = []
        prot_features = []
        
        for i in range(self.num_layers):
            comp_proj = torch.tanh(self.comp_projections[i](comp_feat))
            prot_proj = torch.tanh(self.prot_projections[i](prot_feat))
            
            comp_bilinear = self.bilinear_layers[i](comp_proj)
            attention_scores = torch.bmm(comp_bilinear, prot_proj.transpose(1, 2))
            
            mask_matrix = torch.bmm(
                comp_mask.view(batch_size, -1, 1).float(),
                prot_mask.view(batch_size, 1, -1).float()
            )
            attention_scores = attention_scores * mask_matrix
            
            comp_context = torch.sum(
                attention_scores.unsqueeze(-1) * prot_proj.unsqueeze(1), dim=2
            )
            prot_context = torch.sum(
                attention_scores.transpose(1, 2).unsqueeze(-1) * comp_proj.unsqueeze(1), dim=2
            )
            
            comp_weights = self._masked_softmax(
                self.comp_attention[i](comp_context).squeeze(-1), comp_mask
            )
            prot_weights = self._masked_softmax(
                self.prot_attention[i](prot_context).squeeze(-1), prot_mask
            )
            
            comp_pooled = torch.sum(comp_proj * comp_weights.unsqueeze(-1), dim=1)
            prot_pooled = torch.sum(prot_proj * prot_weights.unsqueeze(-1), dim=1)
            
            comp_features.append(comp_pooled)
            prot_features.append(prot_pooled)
        
        comp_final = self.comp_combine(torch.cat(comp_features, dim=1))
        prot_final = self.prot_combine(torch.cat(prot_features, dim=1))
        
        return comp_final, prot_final, comp_context, prot_context
    
    def _masked_softmax(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        scores_exp = torch.exp(scores - scores_max)
        scores_exp = scores_exp * mask.float()
        return scores_exp / (torch.sum(scores_exp, dim=-1, keepdim=True) + 1e-6)


# ============================
# 主要的多模态融合模块
# ============================

class MultiModalFusion(nn.Module):
    """
    多模态融合模块
    支持多种融合策略：双向注意力、交叉注意力、协同注意力、多头交叉注意力、双线性注意力网络
    """
    
    SUPPORTED_FUSION_TYPES = [
        'bidirectional', 'cross_attention', 'co_attention', 
        'multi_head_cross', 'ban'
    ]
    
    def __init__(self, fusion_type: str, latent_dim: int, num_heads: int = 3, 
                 dropout: float = 0.1, num_layers: int = 4):
        super(MultiModalFusion, self).__init__()
        
        if fusion_type not in self.SUPPORTED_FUSION_TYPES:
            raise ValueError(f"Unsupported fusion type: {fusion_type}. "
                           f"Supported types: {self.SUPPORTED_FUSION_TYPES}")
        
        self.fusion_type = fusion_type
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # 根据融合类型初始化相应模块
        self._init_fusion_modules(latent_dim, num_heads, dropout, num_layers)
        
    def _init_fusion_modules(self, latent_dim: int, num_heads: int, 
                           dropout: float, num_layers: int):
        """根据融合类型初始化相应的模块"""
        
        if self.fusion_type == 'bidirectional':
            self.bidirectional_module = BidirectionalAttentionModule(latent_dim, num_layers)
            
        elif self.fusion_type == 'cross_attention':
            self.cross_attn_c2p = CrossAttentionLayer(latent_dim, num_heads, dropout)
            self.cross_attn_p2c = CrossAttentionLayer(latent_dim, num_heads, dropout)
            self.layer_norm_comp = nn.LayerNorm(latent_dim)
            self.layer_norm_prot = nn.LayerNorm(latent_dim)
            
        elif self.fusion_type == 'co_attention':
            self.co_attention = CoAttentionLayer(latent_dim, dropout)
            self.gated_fusion = GatedFusionLayer(latent_dim)
            
        elif self.fusion_type == 'multi_head_cross':
            self.multi_cross_c2p = nn.ModuleList([
                CrossAttentionLayer(latent_dim, 1, dropout) for _ in range(num_heads)
            ])
            self.multi_cross_p2c = nn.ModuleList([
                CrossAttentionLayer(latent_dim, 1, dropout) for _ in range(num_heads)
            ])
            self.combine_heads_comp = nn.Linear(latent_dim * num_heads, latent_dim)
            self.combine_heads_prot = nn.Linear(latent_dim * num_heads, latent_dim)
            
        elif self.fusion_type == 'ban':
            self.ban_module = BilinearAttentionNetwork(
                latent_dim, latent_dim, latent_dim, num_layers
            )
    
    def forward(self, comp_feat: torch.Tensor, comp_mask: torch.Tensor,
                prot_feat: torch.Tensor, prot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            comp_final: 融合后的化合物特征
            prot_final: 融合后的蛋白质特征
        """
        if self.fusion_type == 'bidirectional':
            return self._bidirectional_fusion(comp_feat, comp_mask, prot_feat, prot_mask)
        elif self.fusion_type == 'cross_attention':
            return self._cross_attention_fusion(comp_feat, comp_mask, prot_feat, prot_mask)
        elif self.fusion_type == 'co_attention':
            return self._co_attention_fusion(comp_feat, comp_mask, prot_feat, prot_mask)
        elif self.fusion_type == 'multi_head_cross':
            return self._multi_head_cross_fusion(comp_feat, comp_mask, prot_feat, prot_mask)
        elif self.fusion_type == 'ban':
            return self._ban_fusion(comp_feat, comp_mask, prot_feat, prot_mask)
    
    def _bidirectional_fusion(self, comp_feat, comp_mask, prot_feat, prot_mask):
        """双向注意力融合"""
        comp_final, prot_final, comp_layerwise, prot_layerwise = self.bidirectional_module(comp_feat, comp_mask, \
                                                                                           prot_feat, prot_mask)
        
        return comp_final, prot_final, comp_layerwise, prot_layerwise
    
    def _cross_attention_fusion(self, comp_feat, comp_mask, prot_feat, prot_mask):
        """交叉注意力融合"""
        batch_size = comp_feat.size(0)
        
        comp_prot_mask = self._create_attention_mask(comp_mask, prot_mask)
        prot_comp_mask = comp_prot_mask.transpose(1, 2)
        
        comp_cross, _ = self.cross_attn_c2p(comp_feat, prot_feat, prot_feat, prot_comp_mask)
        prot_cross, _ = self.cross_attn_p2c(prot_feat, comp_feat, comp_feat, comp_prot_mask)
        
        comp_cross = self.layer_norm_comp(comp_cross + comp_feat)
        prot_cross = self.layer_norm_prot(prot_cross + prot_feat)
        
        comp_pooled = self._masked_global_pool(comp_cross, comp_mask)
        prot_pooled = self._masked_global_pool(prot_cross, prot_mask)
        
        return comp_pooled, prot_pooled
    
    def _co_attention_fusion(self, comp_feat, comp_mask, prot_feat, prot_mask):
        """协同注意力融合"""
        batch_size = comp_feat.size(0)
        
        # 协同注意力计算
        comp_attended, prot_attended, _, _ = self.co_attention(
            comp_feat, prot_feat, comp_mask, prot_mask
        )
        
        comp_fused, prot_fused = self.gated_fusion(
            comp_feat, comp_attended, prot_feat, prot_attended
        )
        
        comp_pooled = self._masked_global_pool(comp_fused, comp_mask)
        prot_pooled = self._masked_global_pool(prot_fused, prot_mask)
        
        return comp_pooled, prot_pooled
    
    def _multi_head_cross_fusion(self, comp_feat, comp_mask, prot_feat, prot_mask):
        """多头交叉注意力融合"""
        batch_size = comp_feat.size(0)
        comp_heads = []
        prot_heads = []

        comp_prot_mask = self._create_attention_mask(comp_mask, prot_mask)
        prot_comp_mask = self._create_attention_mask(prot_mask, comp_mask) 

        for cross_c2p, cross_p2c in zip(self.multi_cross_c2p, self.multi_cross_p2c):
            comp_cross, _ = cross_c2p(comp_feat, prot_feat, prot_feat, comp_prot_mask)
            prot_cross, _ = cross_p2c(prot_feat, comp_feat, comp_feat, prot_comp_mask)
            comp_heads.append(comp_cross)
            prot_heads.append(prot_cross)
        
        comp_multi = torch.cat(comp_heads, dim=-1)
        prot_multi = torch.cat(prot_heads, dim=-1)
        
        comp_combined = self.combine_heads_comp(comp_multi)
        prot_combined = self.combine_heads_prot(prot_multi)
        
        comp_pooled = self._masked_global_pool(comp_combined, comp_mask)
        prot_pooled = self._masked_global_pool(prot_combined, prot_mask)
        
        return comp_pooled, prot_pooled
    
    def _ban_fusion(self, comp_feat, comp_mask, prot_feat, prot_mask):
        """双线性注意力网络融合"""
        batch_size = comp_feat.size(0)
        comp_final, prot_final = self.ban_module(comp_feat, comp_mask, prot_feat, prot_mask)
        
        return comp_final, prot_final
    
    def _create_attention_mask(self, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """创建注意力掩码矩阵"""
        return torch.matmul(mask1.unsqueeze(2).float(), mask2.unsqueeze(1).float())
    
    def _masked_global_pool(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """带掩码的全局池化"""
        mask_expanded = mask.unsqueeze(-1).float()
        masked_features = features * mask_expanded
        pooled = torch.sum(masked_features, dim=1) / (torch.sum(mask_expanded, dim=1) + 1e-6)
        return pooled
