import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model.fusion_layers import MultiModalFusion
from model.final_predict_layer_archs import ConcatMLP, HadamardMLP, \
    DiffAbsConcatMLP, BilinearOuterProduct, BilinearProjection, GatedFusion, EnsembleFusion, OuterProductLinear

FINAL_PREDICT_LAYER = {
    "ConcatMLP": ConcatMLP,
    "HadamardMLP": HadamardMLP,
    "DiffAbsConcatMLP": DiffAbsConcatMLP,
    "BilinearOuterProduct": BilinearOuterProduct,
    "BilinearProjection": BilinearProjection,
    "GatedFusion": GatedFusion,
    "EnsembleFusion": EnsembleFusion,
    "OuterProductLinear":OuterProductLinear
}


class FusionCPI(nn.Module):
    def __init__(self, n_atom, params):
        super(FusionCPI, self).__init__()

        comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, latent_dim, num_layers = \
        params.comp_dim, params.prot_dim, params.gat_dim, params.num_head, params.dropout,\
        params.alpha, params.latent_dim, params.num_layers

        # 新增参数：融合方法类型
        self.fusion_type = getattr(params, 'fusion_type', 'bidirectional')  # 默认使用原始方法

        self.embedding_layer_atom = nn.Embedding(n_atom+1, comp_dim)

        self.dropout = dropout
        self.alpha = alpha
        self.ProtEmbedFC = ProtEmbedFC(1152, out_dim=prot_dim, hidden_dim=512, alpha=0.2)

        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_head)]

        for i, layer in enumerate(self.gat_layers):
            self.add_module('gat_layer_{}'.format(i), layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        # 使用新的多模态融合模块
        self.fusion_module = MultiModalFusion(
            fusion_type=self.fusion_type, 
            latent_dim=latent_dim, 
            num_heads=num_head, 
            dropout=dropout,
            num_layers=num_layers
        )

        # 初始化最终预测层
        self.predictor = FINAL_PREDICT_LAYER[params.final_predict_layer](params.latent_dim)


    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj)) # 最后一层GAT输出comp_dim维度的向量
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha) # 线性变换comp_dim到latent_dim维度
        return atoms_vector


    def forward(self, atoms, atoms_mask, adjacency, prot_embed, prot_mask):
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.ProtEmbedFC(prot_embed, prot_mask=prot_mask)

        cf_final, pf_final = self.fusion_module(atoms_vector, atoms_mask, amino_vector, prot_mask)

        prediction = self.predictor(cf_final, pf_final)
        
        return prediction

    def inference(model, atoms, atoms_mask, adjacency, prot_embed, prot_mask, device):
        """
        使用训练好的模型进行推理。
        (atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)
        参数:
            model: 已训练好的 FusionCPI 模型
            atoms, atoms_mask, adjacency, prot_embed, prot_mask: 已经准备好的输入张量
            device: torch.device("cuda" 或 "cpu")

        返回:
            preds: numpy.ndarray, 模型预测结果
        """
        model.eval()
        with torch.no_grad():
            atoms = atoms.to(device)
            atoms_mask = atoms_mask.to(device)
            adjacency = adjacency.to(device)
            prot_embed = prot_embed.to(device)
            prot_mask = prot_mask.to(device)

            preds = model(atoms, atoms_mask, adjacency, prot_embed, prot_mask)

        return preds

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # 将邻接矩阵中非零位置的注意力值保留，其他位置设为负无穷
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(b, N*N, self.out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class ProtEmbedFC(nn.Module):
    """
    把预训练模型输出的蛋白 embedding 投影到与 CNN 输出一致的维度
    """
    def __init__(self, prot_embed_dim_esmc, out_dim=None, hidden_dim=512, alpha=0.2):
        super(ProtEmbedFC, self).__init__()
        self.alpha = alpha

        self.prot_embed_fc = nn.Sequential(
            nn.Linear(prot_embed_dim_esmc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, prot_embed_1, prot_mask=None):
        """
        prot_embed_1: [B, L, prot_embed_dim_esmc]  预训练 embedding
        prot_mask:    [B, L] (可选) 用于后续做 pooling
        """
        # 先通过 MLP 投影
        amino_vector = self.prot_embed_fc(prot_embed_1)   # [B, L, out_dim]
        amino_vector = F.leaky_relu(amino_vector, self.alpha)

        return amino_vector
