import torch.nn as nn
import torch.nn.functional as F

class FusionCPILoss(nn.Module):
    """
    FusionCPI模型的联合损失函数
    """
    def __init__(self, affinity_weight=None, sequence_weight=None, ignore_index=-100):
        super(FusionCPILoss, self).__init__()
        self.affinity_weight = affinity_weight
        self.sequence_weight = sequence_weight
        self.ignore_index = ignore_index

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, prediction, sequence_logits, labels, target_affinity):
        """
        Returns:
            total_loss
            affinity_loss
            sequence_loss
        """
        if target_affinity.dim() == 1:
            target_affinity = target_affinity.unsqueeze(1)
        affinity_loss = self.mse_loss(prediction, target_affinity)
        
        batch_size, seq_len, vocab_size = sequence_logits.shape
        sequence_logits_flat = sequence_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        sequence_loss = self.ce_loss(sequence_logits_flat, labels_flat)

        total_loss = (self.affinity_weight * affinity_loss + 
                     self.sequence_weight * sequence_loss)
        
        return total_loss, affinity_loss, sequence_loss


class AdaptiveFusionCPILoss(FusionCPILoss):
    """
    自适应调整损失权重的版本
    """
    def __init__(self, initial_affinity_weight=None, initial_sequence_weight=None, 
                 adapt_strategy='loss_magnitude', ignore_index=-100):
        super().__init__(initial_affinity_weight, initial_sequence_weight, ignore_index)
        self.initial_affinity_weight = initial_affinity_weight
        self.initial_sequence_weight = initial_sequence_weight
        self.adapt_strategy = adapt_strategy
        
    def forward(self, prediction, sequence_logits, labels, target_affinity):
        total_loss, affinity_loss, sequence_loss = super().forward(
            prediction, sequence_logits, labels, target_affinity
        )
        
        # 根据策略调整权重
        if self.adapt_strategy == 'loss_magnitude':
            affinity_mag = affinity_loss.detach()
            sequence_mag = sequence_loss.detach()
            
            if affinity_mag > sequence_mag * 2:
                # 亲和力损失过大，降低权重
                self.affinity_weight = self.initial_affinity_weight * 0.5
            elif sequence_mag > affinity_mag * 2:
                # 序列损失过大，降低权重  
                self.sequence_weight = self.initial_sequence_weight * 0.5
            else:
                # 恢复原始权重
                self.affinity_weight = self.initial_affinity_weight
                self.sequence_weight = self.initial_sequence_weight
            
            # 重新计算总损失
            total_loss = (self.affinity_weight * affinity_loss + 
                         self.sequence_weight * sequence_loss)
        
        return total_loss, affinity_loss, sequence_loss