import torch
from torch import nn
import torch.nn.functional as F


class CAN_Smiles_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Smiles_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
  
    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape 
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H) 
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H) 
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col) 
        logits = torch.where(mask_pair, logits, logits - inf) 
        alpha = torch.softmax(logits, dim=2) 
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha)) 
        return alpha 

    def apply_heads(self, x, n_heads, n_ch): 
        ''' 将输入张量x的最后一个维度分割为n_heads和n_ch。'''
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s) 


    def group_embeddings(self, x, mask, group_size): 
        ''' 将输入张量x的第二个维度seq_len分组为group_size。
            同时将相应的掩码mask也分组。(过程类似于卷积)'''
        N, L, D = x.shape 
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2) 
        mask_grouped = mask.view(N, groups, group_size).any(dim=2) 
        return x_grouped, mask_grouped


    def forward(self, protein, sm, mask_prot, mask_sm): 
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        sm_grouped, mask_sm_grouped = self.group_embeddings(sm, mask_sm, self.group_size)
        
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_sm = self.apply_heads(self.query_d(sm_grouped), self.num_heads, self.head_size)
        key_sm = self.apply_heads(self.key_d(sm_grouped), self.num_heads, self.head_size)
        value_sm = self.apply_heads(self.value_d(sm_grouped), self.num_heads, self.head_size)

        logits_dp = torch.einsum('blhd, bkhd->blkh', query_sm, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_sm, key_sm)
        
        alpha_dp = self.alpha_logits(logits_dp, mask_sm_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_sm_grouped, mask_sm_grouped)

        sm_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_dd, value_sm).flatten(-2)) / 2

        return sm_embedding   


class CAN_Protein_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Protein_Layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
  
    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape 
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H) 
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H) 
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col) 
        logits = torch.where(mask_pair, logits, logits - inf) 
        alpha = torch.softmax(logits, dim=2) 
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha)) 
        return alpha 

    def apply_heads(self, x, n_heads, n_ch): 
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s) 


    def group_embeddings(self, x, mask, group_size): 
        N, L, D = x.shape 
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2) 
        mask_grouped = mask.view(N, groups, group_size).any(dim=2) 
        return x_grouped, mask_grouped


    def forward(self, protein, sm, mask_prot, mask_sm): 
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        sm_grouped, mask_sm_grouped = self.group_embeddings(sm, mask_sm, self.group_size)
        
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size) 
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        key_sm = self.apply_heads(self.key_d(sm_grouped), self.num_heads, self.head_size)
        value_sm = self.apply_heads(self.value_d(sm_grouped), self.num_heads, self.head_size)

        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot) 
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_sm) 
        
        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped) 
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_sm_grouped) 

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) + 
                   torch.einsum('blkh, bkhd->blhd', alpha_pd, value_sm).flatten(-2)) / 2 

        return prot_embedding   




class CAN_Fusion_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, args):
        super(CAN_Fusion_Layer, self).__init__()
        self.agg_mode = args.agg_mode 
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
  
    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape 
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H) 
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H) 
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col) 
        logits = torch.where(mask_pair, logits, logits - inf) 
        alpha = torch.softmax(logits, dim=2) 
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha)) 
        return alpha 

    def apply_heads(self, x, n_heads, n_ch): 
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s) 


    def group_embeddings(self, x, mask, group_size): 
        N, L, D = x.shape 
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2) 
        mask_grouped = mask.view(N, groups, group_size).any(dim=2) 
        return x_grouped, mask_grouped


    def forward(self, protein, sm, mask_prot, mask_sm): 
        protein_grouped, mask_prot_grouped = self.group_embeddings(protein, mask_prot, self.group_size)
        sm_grouped, mask_sm_grouped = self.group_embeddings(sm, mask_sm, self.group_size)
        
        query_prot = self.apply_heads(self.query_p(protein_grouped), self.num_heads, self.head_size) 
        key_prot = self.apply_heads(self.key_p(protein_grouped), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein_grouped), self.num_heads, self.head_size)

        query_sm = self.apply_heads(self.query_d(sm_grouped), self.num_heads, self.head_size)
        key_sm = self.apply_heads(self.key_d(sm_grouped), self.num_heads, self.head_size)
        value_sm = self.apply_heads(self.value_d(sm_grouped), self.num_heads, self.head_size)

        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot) 
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_sm) 
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_sm, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_sm, key_sm)
        
        alpha_pp = self.alpha_logits(logits_pp, mask_prot_grouped, mask_prot_grouped) 
        alpha_pd = self.alpha_logits(logits_pd, mask_prot_grouped, mask_sm_grouped) 
        alpha_dp = self.alpha_logits(logits_dp, mask_sm_grouped, mask_prot_grouped)
        alpha_dd = self.alpha_logits(logits_dd, mask_sm_grouped, mask_sm_grouped)

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) + 
                   torch.einsum('blkh, bkhd->blhd', alpha_pd, value_sm).flatten(-2)) / 2 
        sm_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha_dd, value_sm).flatten(-2)) / 2


        if self.agg_mode == 'simple_concatenation':
            output_tensor = sm_embedding.repeat(1, 2, 1) 
            joint_embed = torch.cat([prot_embedding, output_tensor], dim=-1)
            joint_embed = joint_embed[:, :900, :]

        elif self.agg_mode == 'attention_based_alignment':
            cross_alpha = torch.einsum('blhd,bkhd->blkh', prot_embedding, sm_embedding)  # [batch, protein_len, sm_len]
            cross_alpha = F.softmax(cross_alpha, dim=-1)
            sm_embedding_aligned = torch.einsum('blkh,bkhd->blhd', cross_alpha, sm_embedding)
            joint_embed = torch.cat([prot_embedding, sm_embedding_aligned], dim=-1)

        elif self.agg_mode == 'cls':
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            sm_embed = sm_embedding[:, 0]  # query : [batch_size, hidden]
            joint_embed = torch.cat([prot_embed, sm_embed], dim=1)

        return joint_embed   
    


class SingleCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, protein, sm, mask_prot, mask_sm):
        prot_residual = protein
        sm_residual = sm

        query_prot = self.apply_heads(self.query_p(protein), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein), self.num_heads, self.head_size)

        query_sm = self.apply_heads(self.query_d(sm), self.num_heads, self.head_size)
        key_sm = self.apply_heads(self.key_d(sm), self.num_heads, self.head_size)
        value_sm = self.apply_heads(self.value_d(sm), self.num_heads, self.head_size)

        logits_pp = torch.einsum('blhd,bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd,bkhd->blkh', query_prot, key_sm)
        logits_dp = torch.einsum('blhd,bkhd->blkh', query_sm, key_prot)
        logits_dd = torch.einsum('blhd,bkhd->blkh', query_sm, key_sm)

        alpha_pp = self.alpha_logits(logits_pp, mask_prot, mask_prot)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot, mask_sm)
        alpha_dp = self.alpha_logits(logits_dp, mask_sm, mask_prot)
        alpha_dd = self.alpha_logits(logits_dd, mask_sm, mask_sm)

        prot_embedding = (torch.einsum('blkh,bkhd->blhd', alpha_pp, value_prot).flatten(-2) + 
                         torch.einsum('blkh,bkhd->blhd', alpha_pd, value_sm).flatten(-2)) / 2
        sm_embedding = (torch.einsum('blkh,bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                       torch.einsum('blkh,bkhd->blhd', alpha_dd, value_sm).flatten(-2)) / 2

        prot_embedding = self.norm1(prot_embedding + prot_residual)
        sm_embedding = self.norm2(sm_embedding + sm_residual)

        return prot_embedding, sm_embedding

    @staticmethod
    def alpha_logits(logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh,bkh->blkh', mask_row, mask_col)
        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        return torch.where(mask_row, alpha, torch.zeros_like(alpha))

    @staticmethod
    def apply_heads(x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)


class MultiLayer_CAN(nn.Module):
    ''' 支持多层交叉注意力的特征融合 '''
    def __init__(self, hidden_dim, num_heads, args):
        super().__init__()
        self.agg_mode = args.agg_mode
        self.group_size = args.group_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = args.num_can_layers

        self.cross_attn_layers = nn.ModuleList([
            SingleCrossAttentionLayer(hidden_dim, num_heads)
            for _ in range(self.num_layers)
        ])

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped

    def forward(self, protein, sm, mask_prot, mask_sm):
        ''' 最终返回 joint_embed, sm_grouped, prot_grouped '''
        prot_grouped, mask_p = self.group_embeddings(protein, mask_prot, self.group_size)
        sm_grouped, mask_s = self.group_embeddings(sm, mask_sm, self.group_size)

        for layer in self.cross_attn_layers:
            prot_grouped, sm_grouped = layer(prot_grouped, sm_grouped, mask_p, mask_s)

        if self.agg_mode == 'simple_concatenation':
            joint_embed = torch.cat([prot_grouped, sm_grouped], dim=1)

        elif self.agg_mode == 'attention_based_alignment':
            cross_alpha = torch.einsum('blhd,bkhd->blkh', 
                prot_grouped.view(*prot_grouped.shape[:-1], self.num_heads, self.head_size),
                sm_grouped.view(*sm_grouped.shape[:-1], self.num_heads, self.head_size))
            cross_alpha = F.softmax(cross_alpha, dim=-1)
            sm_aligned = torch.einsum('blkh,bkhd->blhd', cross_alpha, 
                sm_grouped.view(*sm_grouped.shape[:-1], self.num_heads, self.head_size))
            joint_embed = torch.cat([prot_grouped, sm_aligned.flatten(-2)], dim=-1)

        elif self.agg_mode == 'cls':
            joint_embed = torch.cat([prot_grouped[:, 0], sm_grouped[:, 0]], dim=1)

        return joint_embed, sm_grouped, prot_grouped