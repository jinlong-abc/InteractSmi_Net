import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset
import math
from torch.cuda.amp import GradScaler, autocast

PAD = 0

def evaluate(model, test_loader_2, vocab_size):
    model.eval()
    total_IntSeq_loss, total_affinity_loss = 0, 0
    total_mae, total_rmse, total_r2 = 0, 0, 0
    n = 0
    for b, batch in tqdm(enumerate(test_loader_2), total=len(test_loader_2)): 
        tokenized_sm, mask_sm, prot_embed, prot_mask, IntSeq_idx, IntSeq_mask, IntSeq_target, label = [
            x.cuda() for x in batch
        ]
        
        with torch.no_grad():
            decoder_output, affinity = model(prot_embed, tokenized_sm, prot_mask, mask_sm, IntSeq_idx, IntSeq_mask) 

        bsz = label.size(0)
        IntSeq_loss = F.cross_entropy(decoder_output.view(-1, vocab_size),
                            IntSeq_target.contiguous().view(-1),
                            ignore_index=PAD)
        affinity_loss = F.mse_loss(affinity.view(-1), label.contiguous().view(-1))  

        rmse = math.sqrt(affinity_loss.item())
        mae = torch.mean(torch.abs(affinity - label)).item()
        r2 = 1 - torch.sum((label - affinity)**2).item() / (torch.sum((label - label.mean())**2).item() + 1e-7)

        total_IntSeq_loss += IntSeq_loss.item() * bsz
        total_affinity_loss += affinity_loss.item() * bsz
        total_mae += mae * bsz
        total_rmse += rmse * bsz
        total_r2 += r2 * bsz
        n += bsz

    return {
        "IntSeq_loss": total_IntSeq_loss / n,
        "affinity_loss": total_affinity_loss / n,
        "MAE": total_mae / n,
        "RMSE": total_rmse / n,
        "R2": total_r2 / n
    }