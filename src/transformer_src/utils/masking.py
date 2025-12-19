import torch

def create_pad_mask(seq, pad_idx):
    if isinstance(pad_idx, bool):
        seq = seq.bool()
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    subsequent_mask = torch.triu(
        torch.ones(seq_len, seq_len), diagonal=1
    ).type(torch.uint8)
    return subsequent_mask.unsqueeze(0) == 0

def create_mask(src, tgt, pad_idx):
    src_mask = create_pad_mask(src, pad_idx)
    
    tgt_len = tgt.size(1)
    tgt_sub_mask = create_causal_mask(tgt_len)
    tgt_pad_mask = create_pad_mask(tgt, pad_idx)
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    
    return src_mask, tgt_mask


def create_decoder_self_att_mask(tgt, pad_idx):
    tgt_len =tgt.size(1)
    tgt_sub_mask = create_causal_mask(tgt_len)
    tgt_pad_mask = create_pad_mask(tgt, pad_idx)
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask


def create_decoder_cross_and_self_att_mask(encoder_mask, decoder_mask, pad_idx=0, 
                                          smiles_len=None, sequence_len=None):
    B, tgt_len = decoder_mask.shape
    _, src_len = encoder_mask.shape

    cross_mask = ~encoder_mask.bool()  # [B, src_len]
    cross_mask = cross_mask[:, None, :].expand(B, tgt_len, src_len)  # [B, tgt_len, src_len]

    device = decoder_mask.device

    if smiles_len is not None and sequence_len is not None:
        prefix_len = 1

        causal_mask = torch.zeros(tgt_len, tgt_len, device=device, dtype=torch.bool)
        for i in range(prefix_len, tgt_len):
            causal_mask[i, i+1:] = True
        
    else:
        raise ValueError("smiles_len and sequence_len must be provided for prefix masking.")
    # padding mask
    pad_mask = ~decoder_mask.bool()
    pad_mask = pad_mask[:, None, :].expand(B, tgt_len, tgt_len)

    tgt_mask = causal_mask[None, :, :].expand(B, tgt_len, tgt_len) | pad_mask

    return (~cross_mask).long(), (~tgt_mask).long()




def create_fusion_attention_masks(src_mask, tgt_mask):
    device = src_mask.device
    src_pad_mask = src_mask.unsqueeze(1).unsqueeze(2)
    tgt_len = tgt_mask.size(1)
    cross_mask = src_pad_mask.expand(-1, -1, tgt_len, -1)
    tgt_pad_mask = tgt_mask.unsqueeze(1).unsqueeze(2)

    return cross_mask, tgt_pad_mask