import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion_layers import MultiModalFusion
from model.final_predict_layer_archs import (
    ConcatMLP,
    HadamardMLP,
    DiffAbsConcatMLP,
    BilinearOuterProduct,
    BilinearProjection,
    GatedFusion,
    EnsembleFusion,
    OuterProductLinear,
)
from transformer_src.models.decoder import Decoder
from transformer_src.modules.generator import *
from transformer_src.modules.embeddings import EmbeddingWithPositionalEncoding


FINAL_PREDICT_LAYER = {
    "ConcatMLP": ConcatMLP,
    "HadamardMLP": HadamardMLP,
    "DiffAbsConcatMLP": DiffAbsConcatMLP,
    "BilinearOuterProduct": BilinearOuterProduct,
    "BilinearProjection": BilinearProjection,
    "GatedFusion": GatedFusion,
    "EnsembleFusion": EnsembleFusion,
    "OuterProductLinear": OuterProductLinear,
}


class FusionCPI(nn.Module):
    def __init__(self, n_atom, params, vocab=None):
        super(FusionCPI, self).__init__()

        (
            comp_dim,
            prot_dim,
            gat_dim,
            num_head,
            dropout,
            alpha,
            latent_dim,
            num_layers,
            multitask,
        ) = (
            params.comp_dim,
            params.prot_dim,
            params.gat_dim,
            params.num_head,
            params.dropout,
            params.alpha,
            params.latent_dim,
            params.num_layers,
            params.multitask,
        )
        self.multitask = multitask

        self.fusion_type = params.fusion_type

        self.embedding_layer_atom = nn.Embedding(n_atom + 1, comp_dim)
        self.alpha = alpha
        self.ProtEmbedFC = ProtEmbedFC(
            1152, out_dim=prot_dim, hidden_dim=512, alpha=0.2
        )

        self.gat_layers = nn.ModuleList(
            [
                GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
                for _ in range(num_head)
            ]
        )
        self.gat_out = GATLayer(
            gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False
        )
        self.W_comp = nn.Linear(comp_dim, latent_dim)
        self.fusion_module = MultiModalFusion(
            fusion_type=self.fusion_type,
            latent_dim=latent_dim,
            num_heads=num_head,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.predictor = FINAL_PREDICT_LAYER[params.final_predict_layer](latent_dim)

        if self.multitask:
            self.output = nn.Linear(latent_dim * latent_dim, 1)
            self.vocab = vocab
            self.vocab_size = params.vocab_size
            self.model_depth = params.model_depth

            self.embedwithpos = EmbeddingWithPositionalEncoding(
                self.vocab_size, model_depth=self.model_depth
            )
            self.transformer_decoder = Decoder(
                n_layers=params.decoder_layers,
                n_heads=params.num_decoder_head,
                model_depth=self.model_depth,
                ff_depth=params.ffn_depth,
                dropout=dropout,
            )
            self.generator = Generator(
                model_depth=self.model_depth, vocab_size=self.vocab_size
            )
            self.encoder_proj = nn.Linear(
                params.latent_dim * 2 * params.num_layers, self.model_depth
            )

    def comp_gat(self, atoms, atoms_mask, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat(
            [gat(atoms_vector, adj) for gat in self.gat_layers], dim=2
        )
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def forward(
        self,
        atoms,
        atoms_mask,
        adjacency,
        prot_embed,
        prot_mask,
        Smiles=None,
        Smiles_mask=None,
        FusionSmi=None,
        FusionSmi_mask=None,
        Sequence=None,
        Sequence_mask=None,
    ):

        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.ProtEmbedFC(prot_embed, prot_mask=prot_mask)
        if self.multitask:
            cf_final, pf_final, comp_layerwise, prot_layerwise = self.fusion_module(
                atoms_vector, atoms_mask, amino_vector, prot_mask
            )
        else:
            cf_final, pf_final, comp_layerwise, prot_layerwise = self.fusion_module(
                atoms_vector, atoms_mask, amino_vector, prot_mask
            )
        if not self.multitask:
            return self.predictor(cf_final, pf_final)
        prediction = self.predictor(cf_final, pf_final)

        b = prot_layerwise.shape[0]
        encoder_output = self.encoder_proj(
            prot_layerwise
        )

        eos_token_id = self.vocab["token2id"]["<EOS>"]
        bos_token_id = self.vocab["token2id"]["<BOS>"]
        pad_token_id = self.vocab["token2id"]["<PAD>"]

        FusionSmi_eos, FusionSmi_mask_eos = self.add_eos_token(
            FusionSmi, FusionSmi_mask, pad_token_id, eos_token_id
        )
        target_sequence = torch.cat([Smiles, FusionSmi_eos], dim=1)
        target_mask = torch.cat(
            [Smiles_mask, FusionSmi_mask_eos], dim=1
        )
        
        bos_token = torch.full(
            (b, 1),
            bos_token_id,
            dtype=FusionSmi.dtype,
            device=FusionSmi.device,
        )
        decoder_input_ids = torch.cat([Smiles, bos_token, FusionSmi], dim=1)
        bos_mask = torch.ones(b, 1, dtype=target_mask.dtype, device=target_mask.device)
        decoder_input_mask = torch.cat([Smiles_mask, bos_mask, FusionSmi_mask], dim=1)

        decoder_input_embed = self.embedwithpos(decoder_input_ids)

        cross_mask, tgt_mask = self.create_decoder_cross_and_self_att_mask(
            prot_mask,
            decoder_input_mask,
            pad_idx=pad_token_id,
            smiles_len=Smiles.size(1),
            sequence_len=Sequence.size(1),
        )

        decoder_output = self.transformer_decoder(
            decoder_input_embed, encoder_output, cross_mask, tgt_mask
        )
        sequence_logits = self.generator(decoder_output)

        fusion_start_idx = Smiles.size(1)
        fusionsmi_labels = target_sequence.clone()
        fusionsmi_labels[:, :fusion_start_idx] = -100
        fusionsmi_labels[target_mask == 0] = -100

        return prediction, sequence_logits, fusionsmi_labels

    @torch.no_grad()
    def inference(
        self,
        atoms,
        atoms_mask,
        adjacency,
        prot_embed,
        prot_mask,
        Smiles=None,
        Smiles_mask=None,
        Sequence=None,
        Sequence_mask=None,
        max_len=1000,
    ):
        """
        简化的推理：greedy search
        max_len 表示“总长度上限”（含 BOS+Smiles+Sequence+生成部分）。
        """
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.ProtEmbedFC(prot_embed, prot_mask=prot_mask)
        cf_final, pf_final, comp_layerwise, prot_layerwise = self.fusion_module(
            atoms_vector, atoms_mask, amino_vector, prot_mask
        )
        if not self.multitask:
            return self.predictor(cf_final, pf_final)
        prediction = self.predictor(cf_final, pf_final)

        device = Smiles.device
        b = comp_layerwise.shape[0]
        encoder_output = self.encoder_proj(prot_layerwise)  # [B, S_enc, model_depth]

        bos_token_id = self.vocab["token2id"]["<BOS>"]
        eos_token_id = self.vocab["token2id"]["<EOS>"]
        pad_token_id = self.vocab["token2id"]["<PAD>"]

        bos_token = torch.full((b, 1), bos_token_id, dtype=Smiles.dtype, device=device)
        bos_mask = torch.ones(b, 1, dtype=Smiles_mask.dtype, device=Smiles_mask.device)
        prefix = torch.cat([Smiles, bos_token], dim=1)
        prefix_mask = torch.cat([Smiles_mask, bos_mask], dim=1)

        generated = prefix
        running_mask = prefix_mask
        alive = torch.ones(b, dtype=torch.bool, device=device)

        smiles_len = Smiles.size(1)
        seq_len = Sequence.size(1)

        while generated.size(1) < max_len:
            dec_inp = self.embedwithpos(generated)
            cross_mask, tgt_mask = self.create_decoder_cross_and_self_att_mask(
                prot_mask,
                running_mask,
                pad_idx=pad_token_id,
                smiles_len=smiles_len,
                sequence_len=seq_len,
            )

            dec_out = self.transformer_decoder(
                dec_inp, encoder_output, cross_mask, tgt_mask
            )
            logits_last = self.generator(dec_out[:, -1, :])  # [B, V]
            next_token = torch.argmax(logits_last, dim=-1)  
            next_token = torch.where(
                alive, next_token, torch.full_like(next_token, pad_token_id)
            )

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            ones_step = torch.ones((b, 1), dtype=running_mask.dtype, device=device)
            running_mask = torch.cat([running_mask, ones_step], dim=1)
            alive = alive & (next_token != eos_token_id)
            if not alive.any():
                break

        return prediction, generated

    def add_eos_token(self, FusionSmi, FusionSmi_mask, pad_token_id, eos_token_id):
        """
            FusionSmi_eos (torch.Tensor)
            FusionSmi_mask_eos (torch.Tensor)
        """
        b, L = FusionSmi.shape
        lengths = (FusionSmi != pad_token_id).sum(dim=1)

        FusionSmi_eos = torch.full((b, L + 1), pad_token_id, dtype=FusionSmi.dtype, device=FusionSmi.device)
        FusionSmi_mask_eos = torch.zeros((b, L + 1), dtype=FusionSmi_mask.dtype, device=FusionSmi_mask.device)

        for i, l in enumerate(lengths):
            l = l.item()
            FusionSmi_eos[i, :l] = FusionSmi[i, :l]
            FusionSmi_mask_eos[i, :l] = 1
            FusionSmi_eos[i, l] = eos_token_id
            FusionSmi_mask_eos[i, l] = 1

        return FusionSmi_eos, FusionSmi_mask_eos

    def create_decoder_cross_and_self_att_mask(self, encoder_mask, decoder_mask, pad_idx=0, 
                                            smiles_len=None, sequence_len=None):
        """
        cross_mask: [B, tgt_len, src_len] (True = masked)
        tgt_mask: [B, tgt_len, tgt_len] (True = masked)
        """
        B, tgt_len = decoder_mask.shape
        _, src_len = encoder_mask.shape

        cross_mask = ~encoder_mask.bool()
        cross_mask = cross_mask[:, None, :].expand(B, tgt_len, src_len)
        device = decoder_mask.device
        
        if smiles_len is not None and sequence_len is not None:
            prefix_len = smiles_len + 1  # +1 for BOS token
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1)
            if prefix_len > 0:
                causal_mask[:prefix_len, :prefix_len] = False
            
        else:
            raise ValueError("smiles_len and sequence_len must be provided for prefix masking.")
        # padding mask
        pad_mask = ~decoder_mask.bool()
        pad_mask = pad_mask[:, None, :].expand(B, tgt_len, tgt_len)
        tgt_mask = causal_mask[None, :, :].expand(B, tgt_len, tgt_len) | pad_mask
        return (~cross_mask).long(), (~tgt_mask).long()


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
        attention = torch.where(
            adj > 0, e, zero_vec
        )
        attention = F.softmax(attention, dim=2)
        h_prime = torch.bmm(attention, Wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        b = Wh.size()[0]
        N = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat_interleave(N, dim=0).view(
            b, N * N, self.out_features
        )
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2
        )

        return all_combinations_matrix.view(b, N, N, 2 * self.out_features)


class ProtEmbedFC(nn.Module):
    def __init__(self, prot_embed_dim_esmc, out_dim=None, hidden_dim=512, alpha=0.2):
        super(ProtEmbedFC, self).__init__()
        self.alpha = alpha

        self.prot_embed_fc = nn.Sequential(
            nn.Linear(prot_embed_dim_esmc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, prot_embed_1, prot_mask=None):
        amino_vector = self.prot_embed_fc(prot_embed_1)
        amino_vector = F.leaky_relu(amino_vector, self.alpha)

        return amino_vector
