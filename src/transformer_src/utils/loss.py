import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import itertools

PAD = 0



def multi_candidate_loss(model_output, targets, inter_mask, args):
    """
    只预测一个答案的损失函数。
    计算多个候选目标中的最小损失，并返回用于反向传播的损失和各子损失指标。
    """
    pred_aa_output_1, affinity = model_output 
    pred_aa_types = pred_aa_output_1[0]  # [B, L, V]
    pred_aa_idxs = pred_aa_output_1[1]   # [B, L, V]
    pred_inter_types = pred_aa_output_1[2]  # [B, L, V]
    
    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets  # 每个是 [B, L, 3]

    device = pred_aa_types.device

    label_true = label_true.to(device)  
    inter_types_true = inter_types_true.to(device)
    aa_idxs_true = aa_idxs_true.to(device)
    aa_types_true = aa_types_true.to(device)

    B, L, V = pred_aa_types.shape
    num_candidates = 3
    losses = []
    sub_losses = []

    for i in range(num_candidates):
        aa_types_i = aa_types_true[:, :, i]         # [B, L]
        aa_idxs_i = aa_idxs_true[:, :, i]           # [B, L]
        inter_types_i = inter_types_true[:, :, i]   # [B, L]

        loss_aa_type = F.cross_entropy(pred_aa_types.view(-1, V), aa_types_i.view(-1), ignore_index=PAD)
        loss_aa_idx = F.cross_entropy(pred_aa_idxs.view(-1, V), aa_idxs_i.view(-1), ignore_index=-PAD)
        loss_inter_type = F.cross_entropy(pred_inter_types.view(-1, V), inter_types_i.view(-1), ignore_index=PAD)

        total_interact_loss = loss_aa_type + loss_aa_idx + loss_inter_type

        losses.append(total_interact_loss)
        sub_losses.append((loss_aa_type, loss_aa_idx, loss_inter_type))

    losses_tensor = torch.stack(losses)  # [3]
    min_interact_loss, min_idx = torch.min(losses_tensor, dim=0)

    min_sub_losses = sub_losses[min_idx]
    loss_type, loss_idx, loss_inter = min_sub_losses

    affinity_loss = F.mse_loss(affinity.squeeze(), label_true.float())

    total_loss = min_interact_loss + args.affinity_loss_weight * affinity_loss

    metrics = {
        'total_loss': total_loss,
        'min_interact_loss': min_interact_loss.item(),
        'loss_aa_type': loss_type.item(),
        'loss_aa_idx': loss_idx.item(),
        'loss_inter_type': loss_inter.item(),
        'affinity': affinity_loss.item(),
    }

    return metrics


def evaluate_loss_multi_candidate(model, test_loader, args):
    """完整评估流程
    prot_batch, drug_batch, prot_mask_batch, drug_mask_batch, smiles_list_idx, smiles_list_idx_mask"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    metric_log = {
        'total': 0.0,
        'min_interact_loss': 0.0,  
        'aa_types': 0.0,
        'aa_idxs': 0.0,
        'inter_types': 0.0,
        'affinity': 0.0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            prot, drug, prot_mask, drug_mask, aa_types, aa_idxs, inter_types, inter_mask, label, smiles_list, smiles_list_idx, smiles_list_idx_mask = batch

            prot = prot.to(device)
            drug = drug.to(device)
            prot_mask = prot_mask.to(device)
            drug_mask = drug_mask.to(device)
            label = label.to(device)

            aa_types = aa_types.to(device)
            aa_types = F.pad(aa_types, (0, 0, 0, args.max_Smi_len - aa_types.shape[1]), value=PAD) 
            aa_idxs = aa_idxs.to(device)
            aa_idxs = F.pad(aa_idxs, (0, 0, 0, args.max_Smi_len - aa_idxs.shape[1]), value=PAD) 
            inter_types = inter_types.to(device)
            inter_types = F.pad(inter_types, (0, 0, 0, args.max_Smi_len - inter_types.shape[1]), value=PAD) 
            inter_mask = inter_mask.to(device)
            inter_mask = F.pad(inter_mask, (0, args.max_Smi_len - inter_mask.shape[1]), value=PAD) 

            padded_tensors = []
            for tensor in smiles_list_idx_mask:
                pad_size = args.max_Smi_len - tensor.size(0)
                padded_tensor = F.pad(tensor, (0, pad_size), value=PAD)
                padded_tensors.append(padded_tensor)
            smiles_list_idx_mask = torch.stack(padded_tensors).to(device)

            padded_tensors_2 = []
            for tensor in smiles_list_idx:
                pad_size_2 = args.max_Smi_len - tensor.size(0)
                padded_tensor_2 = F.pad(tensor, (0, pad_size_2), value=PAD)
                padded_tensors_2.append(padded_tensor_2)
            smiles_list_idx = torch.stack(padded_tensors_2).to(device)

            smiles_list_idx = smiles_list_idx.to(device)
            smiles_list_idx_mask = smiles_list_idx_mask.to(device)

            outputs = model(prot, drug, prot_mask, drug_mask, smiles_list_idx, smiles_list_idx_mask)

            losses = multi_candidate_loss_evaluate(
                model_output=outputs,
                targets=(aa_types, aa_idxs, inter_types, label),
                inter_mask=inter_mask,
                args=args
            )

            metric_log['total'] += losses['total_loss']
            metric_log['min_interact_loss'] += losses['min_interact_loss']
            metric_log['aa_types'] += losses['aa_types_loss']
            metric_log['aa_idxs'] += losses['aa_idxs_loss']
            metric_log['inter_types'] += losses['inter_types_loss']
            metric_log['affinity'] += losses['affinity_loss']

    num_batches = len(test_loader)
    return {k: v/num_batches for k, v in metric_log.items()}

def multi_candidate_loss_evaluate(model_output, targets, inter_mask, args):
    pred_aa_output_1, affinity = model_output 
    pred_aa_types = pred_aa_output_1[0]  # [B, L, V]
    pred_aa_idxs = pred_aa_output_1[1]   # [B, L, V]
    pred_inter_types = pred_aa_output_1[2]  # [B, L, V]

    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets  # 每个是 [B, L, 3]

    device = pred_aa_types.device

    label_true = label_true.to(device)  
    inter_types_true = inter_types_true.to(device)
    aa_idxs_true = aa_idxs_true.to(device)
    aa_types_true = aa_types_true.to(device)

    B, L, V = pred_aa_types.shape
    num_candidates = 3
    losses = []
    sub_losses = []

    for i in range(num_candidates):
        aa_types_i = aa_types_true[:, :, i]
        aa_idxs_i = aa_idxs_true[:, :, i]
        inter_types_i = inter_types_true[:, :, i]

        loss_aa_type = F.cross_entropy(pred_aa_types.view(-1, V), aa_types_i.view(-1), ignore_index=PAD)
        loss_aa_idx = F.cross_entropy(pred_aa_idxs.view(-1, V), aa_idxs_i.view(-1), ignore_index=PAD)
        loss_inter_type = F.cross_entropy(pred_inter_types.view(-1, V), inter_types_i.view(-1), ignore_index=PAD)

        total_interact_loss = loss_aa_type + loss_aa_idx + loss_inter_type

        losses.append(total_interact_loss)
        sub_losses.append((loss_aa_type, loss_aa_idx, loss_inter_type))

    losses_tensor = torch.stack(losses)  # [3]
    min_interact_loss, min_idx = torch.min(losses_tensor, dim=0)

    min_sub_losses = sub_losses[min_idx]
    loss_type, loss_idx, loss_inter = min_sub_losses

    affinity_loss = F.mse_loss(affinity.squeeze(), label_true.float())

    total_loss = min_interact_loss + args.affinity_loss_weight * affinity_loss

    return {
        'total_loss': total_loss.item(),
        'min_interact_loss': min_interact_loss.item(),
        'aa_types_loss': loss_type.item(),
        'aa_idxs_loss': loss_idx.item(),
        'inter_types_loss': loss_inter.item(),
        'affinity_loss': affinity_loss.item(),
    }


def evaluate_v2(model, test_loader, args):
    """完整评估流程
    prot_batch, drug_batch, prot_mask_batch, drug_mask_batch, smiles_list_idx, smiles_list_idx_mask"""
    model.eval()
    metric_log = {
        'total': 0.0,
        'aa_types': 0.0,
        'aa_idxs': 0.0,
        'inter_types': 0.0,
        'affinity': 0.0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            # 数据预处理（保持原有逻辑）
            # prot_batch, drug_batch, prot_mask_batch, drug_mask_batch, aa_types_batch, aa_idxs_batch, inter_types_batch, inter_mask_batch, label_batch, smiles_list, smiles_list_idx, smiles_list_idx_mask
            prot, drug, prot_mask, drug_mask, aa_types, aa_idxs, inter_types, inter_mask, label, smiles_list, smiles_list_idx, smiles_list_idx_mask = batch
            device = prot.device
            # 设备转移
            prot = prot.to(device)
            drug = drug.to(device)
            prot_mask = prot_mask.to(device)
            drug_mask = drug_mask.to(device)
            aa_types = aa_types.to(device)
            aa_idxs = aa_idxs.to(device)
            inter_types = inter_types.to(device)
            inter_mask = inter_mask.to(device)
            label = label.to(device)

            # 将aa_types_batch等填充为序列长度100，用0填充
            aa_types = aa_types.cuda() # aa_types_batch: torch.Size([32, 81, 3])
            aa_types = F.pad(aa_types, (0, 0, 0, args.max_Smi_len - aa_types.shape[1]), value=PAD) # 填充到120
            aa_idxs = aa_idxs.cuda()
            aa_idxs = F.pad(aa_idxs, (0, 0, 0, args.max_Smi_len - aa_idxs.shape[1]), value=PAD) # 填充到120
            inter_types = inter_types.cuda()
            inter_types = F.pad(inter_types, (0, 0, 0, args.max_Smi_len - inter_types.shape[1]), value=PAD) # 填充到120
            inter_mask = inter_mask.cuda() # inter_mask_batch: torch.Size([32, 81])
            inter_mask = F.pad(inter_mask, (0, args.max_Smi_len - inter_mask.shape[1]), value=PAD) # 填充到120
            
            # 先对每个张量进行填充，再堆叠
            padded_tensors = []
            for tensor in smiles_list_idx_mask:
                # 计算需要填充的长度
                pad_size = args.max_Smi_len - tensor.size(0)
                # 在末尾填充（右侧填充）
                padded_tensor = F.pad(tensor, (0, pad_size), value=PAD)
                padded_tensors.append(padded_tensor)
            # 现在所有张量长度一致，可以安全堆叠
            smiles_list_idx_mask = torch.stack(padded_tensors).to(device)

            padded_tensors_2 = []
            for tensor in smiles_list_idx:
                # 计算需要填充的长度
                pad_size_2 = args.max_Smi_len - tensor.size(0)
                # 在末尾填充（右侧填充）
                padded_tensor_2 = F.pad(tensor, (0, pad_size_2), value=PAD)
                padded_tensors_2.append(padded_tensor_2)
            # 现在所有张量长度一致，可以安全堆叠
            smiles_list_idx = torch.stack(padded_tensors_2).cuda()

            smiles_list_idx = smiles_list_idx.cuda() # [32, 120]
            smiles_list_idx_mask = smiles_list_idx_mask.to(device) # [32, 120]

            # 前向传播
            outputs = model(prot, drug, prot_mask, drug_mask, smiles_list_idx, smiles_list_idx_mask)  # 根据实际参数调整
            
            # 计算损失
            losses = evaluate_loss_v2(
                model_output=outputs,
                targets=(aa_types, aa_idxs, inter_types, label),
                inter_mask=inter_mask,
                args=args
            )
            
            # 累加指标
            metric_log['total'] += losses['total_loss']
            metric_log['aa_types'] += losses['aa_types_loss']
            metric_log['aa_idxs'] += losses['aa_idxs_loss']
            metric_log['inter_types'] += losses['inter_types_loss']
            metric_log['affinity'] += losses['affinity_loss']
    
    # 计算平均指标
    num_batches = len(test_loader)
    return {k: v/num_batches for k, v in metric_log.items()}



def evaluate_loss_v2(model_output, targets, inter_mask, args):
    """
    评估专用损失函数，与compute_loss_disorder_v2保持相同动态权重逻辑
    """
    # 解包模型输出
    aa_types_pred, aa_idxs_pred, inter_types_pred, affinity_pred = model_output
    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets # torch.Size([B, L, 3])

    # 生成所有排列组合 (3! = 6种)
    perms = list(itertools.permutations(range(3))) 
    batch_size = aa_types_pred[0].size(0) # ([B,L,V], [B,L,V], [B,L,V])
    device = aa_types_pred[0].device # 设备

    # inter_types_true = inter_types_true.to(device)  # 确保标签在正确的设备上    
    label_true = label_true.to(device)  # 确保标签在正确的设备上

    # 初始化最小损失
    min_aa_loss = torch.full((batch_size,), float('inf'), device=device) # [B] 其中所有值都是inf
    min_idx_loss = torch.full((batch_size,), float('inf'), device=device)
    min_inter_loss = torch.full((batch_size,), float('inf'), device=device)

    # 概率计算辅助函数
    def get_probs(pred, true): # get_probs([B,L,V], [B,L])
        probs = torch.softmax(pred, dim=-1) # [B, seq, vocab]的最后一维词表上做softmax，形状不变
        true_probs = torch.gather(probs, -1, true.unsqueeze(-1)).squeeze(-1) # gather([B, seq, vocab], -1, [B, seq, 1])
        return true_probs  # [B, seq] # 每个正确位置对应的预测概率

    # 遍历所有排列组合
    for perm in perms:
        # 对齐预测头
        aligned_aa = [aa_types_pred[i] for i in perm] # ([B,L,V]_2, [B,L,V]_1, [B,L,V]_0)
        aligned_idx = [aa_idxs_pred[i] for i in perm]
        aligned_inter = [inter_types_pred[i] for i in perm]

        # 计算各属性正确概率 # ...表示保留前面所有维度，只在最后一维取出第i个元素
        aa_probs = torch.stack( # p：[B,L,V], aa_types_true[..., i]: [B,L]
            [get_probs(p, aa_types_true[..., i]) for i, p in zip(range(3), aligned_aa)],
            dim=1
        )  # [[B, seq], [B, seq], [B, seq]] ==> [B, 3, seq] ==> 3个预测头的概率
        
        idx_probs = torch.stack(
            [get_probs(p, aa_idxs_true[..., i]) for i, p in zip(range(3), aligned_idx)],
            dim=1
        )
        
        inter_probs = torch.stack(
            [get_probs(p, inter_types_true[..., i]) for i, p in zip(range(3), aligned_inter)],
            dim=1
        )

        # 动态权重计算
        p_all = aa_probs * idx_probs * inter_probs # [B, 3, seq] 逐元素相乘形状不变
        p_type_idx = aa_probs * idx_probs * (1 - inter_probs)
        p_type_inter = aa_probs * (1 - idx_probs) * inter_probs
        weights = 1.0*p_all + 0.66*p_type_idx + 0.33*p_type_inter  # [B, 3, seq]

        # 计算原始交叉熵损失
        aa_loss = sum( # permute维度重排 从[B, seq, vocab]到[B, vocab, seq]
            F.cross_entropy(p.permute(0,2,1), aa_types_true[..., i], reduction='none') # aa_types_true[..., i]： 取最后一维第i个元素形状由[B, seq, 3]变成[B, seq]
            for i, p in zip(perm, aligned_aa)
        )  # 三个[B, seq]求和 结果仍然是[B, seq]代表每个位置的损失
        
        idx_loss = sum(
            F.cross_entropy(p.permute(0,2,1), aa_idxs_true[..., i], reduction='none')
            for i, p in zip(perm, aligned_idx)
        )
        
        inter_loss = sum(
            F.cross_entropy(p.permute(0,2,1), inter_types_true[..., i], reduction='none')
            for i, p in zip(perm, aligned_inter)
        )

        # 调整维度以正确广播
        inter_mask_expanded = inter_mask.unsqueeze(1)  # [B, 1, seq]
        weights = weights * inter_mask_expanded  # 确保权重在填充位置为0

        # 应用动态权重，调整维度
        aa_loss = aa_loss.unsqueeze(1) * (1 - weights)  # [B, seq，1] * [B, 3, seq]
        aa_loss = aa_loss.sum(dim=(1, 2)) / (inter_mask.sum(dim=1) + 1e-8)

        idx_loss = idx_loss.unsqueeze(1) * (1 - weights)
        idx_loss = idx_loss.sum(dim=(1, 2)) / (inter_mask.sum(dim=1) + 1e-8)

        inter_loss = inter_loss.unsqueeze(1) * (1 - weights)
        inter_loss = inter_loss.sum(dim=(1, 2)) / (inter_mask.sum(dim=1) + 1e-8)
        # # 应用动态权重和mask
        # aa_loss = (aa_loss.unsqueeze(1) * (1 - weights) * inter_mask.unsqueeze(1)).sum(dim=(1,2)) 
        # aa_loss = aa_loss / (inter_mask.sum(dim=1) + 1e-8)
        
        # idx_loss = (idx_loss.unsqueeze(1) * (1 - weights) * inter_mask.unsqueeze(1)).sum(dim=(1,2))
        # idx_loss = idx_loss / (inter_mask.sum(dim=1) + 1e-8)
        
        # inter_loss = (inter_loss.unsqueeze(1) * (1 - weights) * inter_mask.unsqueeze(1)).sum(dim=(1,2))
        # inter_loss = inter_loss / (inter_mask.sum(dim=1) + 1e-8)

        # 保留最小损失
        min_aa_loss = torch.minimum(min_aa_loss, aa_loss)
        min_idx_loss = torch.minimum(min_idx_loss, idx_loss)
        min_inter_loss = torch.minimum(min_inter_loss, inter_loss)

    # 计算平均损失
    aa_loss = min_aa_loss.mean()
    idx_loss = min_idx_loss.mean()
    inter_loss = min_inter_loss.mean()

    # 亲和力损失
    affinity_loss = F.mse_loss(affinity_pred.squeeze(), label_true.float())

    # 总损失
    total_loss = aa_loss + idx_loss + inter_loss + args.affinity_loss_weight * affinity_loss

    return {
        'total_loss': total_loss.item(),
        'aa_types_loss': aa_loss.item(),
        'aa_idxs_loss': idx_loss.item(),
        'inter_types_loss': inter_loss.item(),
        'affinity_loss': affinity_loss.item()
    }


def compute_loss_disorder_v2(model_output, targets, inter_mask, args):
    """
    改进版动态权重损失函数，基于概率的连续权重调整
    """
    # 解包模型输出 aa_types_output = (aa_types_output_1 [B, seq, vocab], aa_types_output_2 [B, seq, vocab], aa_types_output_3 [B, seq, vocab])都是元组
    aa_types_pred, aa_idxs_pred, inter_types_pred, affinity_pred = model_output
    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets # [B, seq, 3]
    
    # 生成所有排列组合 (3! = 6种)
    perms = list(itertools.permutations(range(3))) # [6, 3]
    batch_size = aa_types_pred[0].size(0)
    device = aa_types_pred[0].device

    # 计算各属性概率分布
    def get_probs(pred, true): # pred：[B, seq, vocab]， true：[B, seq]
        probs = torch.softmax(pred, dim=-1)
        true_probs = torch.gather(probs, -1, true.unsqueeze(-1)).squeeze(-1)
        return true_probs  # [B, seq]

    # 初始化最小损失
    min_aa_loss = torch.zeros(batch_size, device=device)
    min_idx_loss = torch.zeros(batch_size, device=device)
    min_inter_loss = torch.zeros(batch_size, device=device)
    min_aa_loss[:] = float('inf')
    min_idx_loss[:] = float('inf')
    min_inter_loss[:] = float('inf')

    # 遍历所有排列组合
    for perm in itertools.permutations(range(3)): # [6, 3]
        # 对齐预测与真实标签，aligned_aa 是一个包含三个张量的列表，其中每个张量的形状为 [B, seq, vocab]
        aligned_aa = [aa_types_pred[i] for i in perm] # 按照六种排列组合，从新排序顺序,例如perm是(1,2,3) ==> ([B, seq, vocab]-1, [B, seq, vocab]-2, [B, seq, vocab]-3)
        aligned_idx = [aa_idxs_pred[i] for i in perm]
        aligned_inter = [inter_types_pred[i] for i in perm]

        # 计算各属性正确概率最后维度==>[B, 3, seq]
        aa_probs = torch.stack([get_probs(p, t[..., i]) for i,p,t in zip(perm, aligned_aa, [aa_types_true]*3)], dim=1) # [B, seq, 3]*3 ==> [B, seq, 3, 3] ==> t[..., i]变成了[B, seq, 3]
        idx_probs = torch.stack([get_probs(p, t[..., i]) for i,p,t in zip(perm, aligned_idx, [aa_idxs_true]*3)], dim=1)
        inter_probs = torch.stack([get_probs(p, t[..., i]) for i,p,t in zip(perm, aligned_inter, [inter_types_true]*3)], dim=1)

        # 计算组合权重（保持梯度）==> 逐元素点乘最后形状还是 [B, 3, seq]
        p_all = aa_probs * idx_probs * inter_probs
        p_type_idx = aa_probs * idx_probs * (1 - inter_probs)
        p_type_inter = aa_probs * (1 - idx_probs) * inter_probs
        weights = 1.0*p_all + 0.66*p_type_idx + 0.33*p_type_inter  # [B, 3, seq]

        # ========================调试
        # for i,p,t in zip(perm, aligned_aa, [aa_types_true]*3):
        #     print(f"p{p.shape}, t{t[...,i].shape}") # [B, seq, vocab] [B, seq] ==> ptorch.Size([32, 100, 245]), ttorch.Size([32, 81])

        aa_loss = sum(F.cross_entropy(p.permute(0,2,1), t[...,i], reduction='none') 
                   for i,p,t in zip(perm, aligned_aa, [aa_types_true]*3))
        idx_loss = sum(F.cross_entropy(p.permute(0,2,1), t[...,i], reduction='none') 
                   for i,p,t in zip(perm, aligned_idx, [aa_idxs_true]*3))
        inter_loss = sum(F.cross_entropy(p.permute(0,2,1), t[...,i], reduction='none') 
                   for i,p,t in zip(perm, aligned_inter, [inter_types_true]*3))

        # 调整维度以正确广播
        inter_mask_expanded = inter_mask.unsqueeze(1)  # [B, 1, seq]
        weights = weights * inter_mask_expanded  # 确保权重在填充位置为0

        # 应用动态权重，调整维度
        aa_loss = aa_loss.unsqueeze(1) * (1 - weights)  # [B, 3, seq]
        aa_loss = aa_loss.sum(dim=(1, 2)) / (inter_mask.sum(dim=1) + 1e-8)

        idx_loss = idx_loss.unsqueeze(1) * (1 - weights)
        idx_loss = idx_loss.sum(dim=(1, 2)) / (inter_mask.sum(dim=1) + 1e-8)

        inter_loss = inter_loss.unsqueeze(1) * (1 - weights)
        inter_loss = inter_loss.sum(dim=(1, 2)) / (inter_mask.sum(dim=1) + 1e-8)
        # # 应用动态权重
        # aa_loss = (aa_loss * (1 - weights) * inter_mask).sum(dim=1) / (inter_mask.sum(dim=1)+1e-8) # (1 - weights) * inter_mask 触发广播机制 [B, 3, seq]
        # idx_loss = (idx_loss * (1 - weights) * inter_mask).sum(dim=1) / (inter_mask.sum(dim=1)+1e-8)
        # inter_loss = (inter_loss * (1 - weights) * inter_mask).sum(dim=1) / (inter_mask.sum(dim=1)+1e-8)

        # 保留最小损失，最后形状是 [B]
        min_aa_loss = torch.minimum(min_aa_loss, aa_loss) # 
        min_idx_loss = torch.minimum(min_idx_loss, idx_loss)
        min_inter_loss = torch.minimum(min_inter_loss, inter_loss)

    # 计算最终损失，对Batch求平均，最后形状是标量
    aa_loss = min_aa_loss.mean()
    idx_loss = min_idx_loss.mean()
    inter_loss = min_inter_loss.mean()

    # 亲和力回归损失
    affinity_loss = F.mse_loss(affinity_pred.squeeze(), label_true.float())

    total_loss = aa_loss + idx_loss + inter_loss + args.affinity_loss_weight * affinity_loss
    return total_loss, (aa_loss.item(), idx_loss.item(), inter_loss.item(), affinity_loss.item())

# #######################################################################以上为动态权重损失函数V2############################################################

def compute_loss_disorder(model_output, targets, inter_mask, args):
    """
    考虑顺序可以无序的情况
    复合损失函数，处理三类相互作用预测和一个回归任务
    Args:
        model_output (tuple): 模型输出 (aa_types_output, aa_idxs_output, inter_types_output, affinity)
        targets (tuple): 真实标签 (aa_types, aa_idxs, inter_types, label)
        inter_mask (Tensor): 相互作用掩码 [batch_size, seq_len]
    
    Returns:
        tuple: (总损失, 各分项损失)
    """
    # 解包模型输出 aa_types_output = (aa_types_output_1 [B, seq, vocab], aa_types_output_2 [B, seq, vocab], aa_types_output_3 [B, seq, vocab])都是元组
    aa_types_pred, aa_idxs_pred, inter_types_pred, affinity_pred = model_output
    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets
    
    # 初始化各分项损失
    aa_types_loss = 0.0
    aa_idxs_loss = 0.0
    inter_types_loss = 0.0
    
    # 生成所有可能的排列组合 (3! = 6种)
    perms = list(itertools.permutations(range(3)))
    perms = torch.tensor(perms, dtype=torch.long)  # [6, 3]

    # 将预测结果堆叠为张量以便索引
    aa_preds = torch.stack(aa_types_pred, dim=0)  # [3, B, seq, vocab] 将元组堆叠成张量
    idx_preds = torch.stack(aa_idxs_pred, dim=0)  # [3, B, seq, vocab]
    inter_preds = torch.stack(inter_types_pred, dim=0)  # [3, B, seq, vocab]

    # 将真实标签拆分为三个独立的交互
    aa_targets = torch.stack([aa_types_true[..., i] for i in range(3)], dim=0)  # [3, B, seq]
    idx_targets = torch.stack([aa_idxs_true[..., i] for i in range(3)], dim=0)  # [3, B, seq]
    inter_targets = torch.stack([inter_types_true[..., i] for i in range(3)], dim=0)  # [3, B, seq]

    # 计算每个排列的损失
    batch_size = aa_preds.shape[1]
    device = aa_preds.device

    # 初始化损失矩阵 [6 permutations, batch_size]
    total_aa_loss = torch.zeros(len(perms), batch_size, device=device)
    total_idx_loss = torch.zeros(len(perms), batch_size, device=device)
    total_inter_loss = torch.zeros(len(perms), batch_size, device=device)

    for perm_idx, perm in enumerate(perms): # 1，[0,1,2]
        # 按当前排列重新排列预测结果
        perm_aa = aa_preds[perm]  # [3, B, seq, vocab]
        perm_idx = idx_preds[perm]  # [3, B, seq, vocab]
        perm_inter = inter_preds[perm]  # [3, B, seq, vocab]

        # 计算三个位置的损失
        for i in range(3):
            # 氨基酸类型损失
            aa_loss = F.cross_entropy(
                perm_aa[i].permute(0, 2, 1),  # [B, vocab, seq]
                aa_targets[i],  # [B, seq]
                reduction='none'
            )  # [B, seq]
            aa_loss = (aa_loss * inter_mask).sum(dim=1) / (inter_mask.sum(dim=1) + 1e-8)
            total_aa_loss[perm_idx] += aa_loss

            # 氨基酸索引损失
            idx_loss = F.cross_entropy(
                perm_idx[i].permute(0, 2, 1),
                idx_targets[i],
                reduction='none'
            )
            idx_loss = (idx_loss * inter_mask).sum(dim=1) / (inter_mask.sum(dim=1) + 1e-8)
            total_idx_loss[perm_idx] += idx_loss

            # 交互类型损失
            inter_loss = F.cross_entropy(
                perm_inter[i].permute(0, 2, 1),
                inter_targets[i],
                reduction='none'
            )
            inter_loss = (inter_loss * inter_mask).sum(dim=1) / (inter_mask.sum(dim=1) + 1e-8)
            total_inter_loss[perm_idx] += inter_loss

    # 取每个样本的最小损失
    aa_types_loss = total_aa_loss.min(dim=0)[0].mean()
    aa_idxs_loss = total_idx_loss.min(dim=0)[0].mean()
    inter_types_loss = total_inter_loss.min(dim=0)[0].mean()

    # 计算亲和力回归损失
    affinity_loss = F.mse_loss(affinity_pred.squeeze(), label_true.float())

    # 总损失
    total_loss = args.aa_type_weight * aa_types_loss + args.aa_idx_weight * aa_idxs_loss + args.inter_type_weight * inter_types_loss + args.affinity_weight * affinity_loss

    return (total_loss, (aa_types_loss.item(), aa_idxs_loss.item(), inter_types_loss.item(), affinity_loss.item()))




#############################################################以下损失函数未考虑顺序可变性###########################################################
def compute_loss(model_output, targets, inter_mask, args):
    """
    parser.add_argument('--affinity_weight', type=float, default=0.5, help='weight of the loss function')
    parser.add_argument('--aa_type_weight', type=float, default=0.4, help='weight of the loss function')
    parser.add_argument('--aa_idx_weight', type=float, default=0.4, help='weight of the loss function')
    parser.add_argument('--inter_type_weight', type=float, default=0.2, help='weight of the loss function')
    复合损失函数，处理三类相互作用预测和一个回归任务
    Args:
        model_output (tuple): 模型输出 (aa_types_output, aa_idxs_output, inter_types_output, affinity)
        targets (tuple): 真实标签 (aa_types, aa_idxs, inter_types, label)
        inter_mask (Tensor): 相互作用掩码 [batch_size, seq_len]
    
    Returns:
        tuple: (总损失, 各分项损失)
    """
    # 解包模型输出和真实标签
    aa_types_pred, aa_idxs_pred, inter_types_pred, affinity_pred = model_output
    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets
    
    # 初始化各分项损失
    aa_types_loss = 0.0
    aa_idxs_loss = 0.0
    inter_types_loss = 0.0
    
    # 计算三类相互作用的交叉熵损失（每个类型三个预测头）
    for i in range(3):
        # 氨基酸类型损失
        aa_type_pred = aa_types_pred[i].permute(0, 2, 1)  # [B, vocab, seq]
        aa_types_loss += masked_cross_entropy(aa_type_pred, aa_types_true[..., i], inter_mask)
        
        # 氨基酸索引损失
        aa_idx_pred = aa_idxs_pred[i].permute(0, 2, 1)
        aa_idxs_loss += masked_cross_entropy(aa_idx_pred, aa_idxs_true[..., i], inter_mask)
        
        # 相互作用类型损失
        inter_type_pred = inter_types_pred[i].permute(0, 2, 1)
        inter_types_loss += masked_cross_entropy(inter_type_pred, inter_types_true[..., i], inter_mask)
    
    # 计算亲和力回归损失
    affinity_loss = F.mse_loss(affinity_pred.squeeze(-1), label_true)
    
    # 总损失
    total_loss = args.aa_type_weight * aa_types_loss + args.aa_idx_weight * aa_idxs_loss + args.inter_type_weight * inter_types_loss + args.affinity_weight * affinity_loss
    
    return (
        total_loss,
        (aa_types_loss.detach(), aa_idxs_loss.detach(), inter_types_loss.detach(), affinity_loss.detach())
    )

def masked_cross_entropy(pred, target, mask):
    """
    带掩码的交叉熵损失
    Args:
        pred (Tensor): [batch_size, num_classes, seq_len]
        target (Tensor): [batch_size, seq_len]
        mask (Tensor): [batch_size, seq_len]
    """
    # 计算逐位置交叉熵
    loss = F.cross_entropy(pred, target, reduction='none')  # [B, seq_len]
    
    # 应用掩码
    masked_loss = (loss * mask).sum()
    
    # 归一化（避免除以零）
    valid = mask.sum()
    return masked_loss / (valid + 1e-8)

#===================================================================以下为评估用损失函数=====================================================

def evaluate_loss(model_output, targets, inter_mask, args):
    """
    评估专用损失函数（与训练用compute_loss保持相同计算逻辑）
    参数结构与compute_loss完全一致，方便代码复用
    """
    # 解包模型输出和真实标签
    aa_types_pred, aa_idxs_pred, inter_types_pred, affinity_pred = model_output
    aa_types_true, aa_idxs_true, inter_types_true, label_true = targets
    # [32, 72, 3]

    # 初始化各分项损失
    aa_types_loss = 0.0
    aa_idxs_loss = 0.0
    inter_types_loss = 0.0
    
    # 计算三类相互作用的交叉熵损失（每个类型三个预测头）
    for i in range(3):
        # 氨基酸类型损失（保持与训练时相同的permute操作）
        aa_type_pred = aa_types_pred[i].permute(0, 2, 1)
        aa_types_loss += masked_cross_entropy(aa_type_pred, aa_types_true[..., i], inter_mask)
        
        # 氨基酸索引损失（修正：使用交叉熵而非MSE）
        aa_idx_pred = aa_idxs_pred[i].permute(0, 2, 1)
        aa_idxs_loss += masked_cross_entropy(aa_idx_pred, aa_idxs_true[..., i], inter_mask)
        
        # 相互作用类型损失
        inter_type_pred = inter_types_pred[i].permute(0, 2, 1)
        inter_types_loss += masked_cross_entropy(inter_type_pred, inter_types_true[..., i], inter_mask)
    
    # 亲和力回归损失（保持与训练一致）
    affinity_loss = F.mse_loss(affinity_pred.squeeze(-1), label_true)
    
    # 加权总损失（使用args中的权重参数）
    total_loss = (
        args.aa_type_weight * aa_types_loss + 
        args.aa_idx_weight * aa_idxs_loss + 
        args.inter_type_weight * inter_types_loss + 
        args.affinity_weight * affinity_loss
    )
    
    # 返回所有损失项（不保留梯度）
    return {
        'total_loss': total_loss.item(),
        'aa_types_loss': aa_types_loss.item(),
        'aa_idxs_loss': aa_idxs_loss.item(),
        'inter_types_loss': inter_types_loss.item(),
        'affinity_loss': affinity_loss.item()
    }


def evaluate(model, test_loader, args):
    """完整评估流程"""
    model.eval()
    metric_log = {
        'total': 0.0,
        'aa_types': 0.0,
        'aa_idxs': 0.0,
        'inter_types': 0.0,
        'affinity': 0.0
    }
    
    with torch.no_grad():
        for batch in test_loader:
            # 数据预处理（保持原有逻辑）
            prot, drug, prot_mask, drug_mask, aa_types, aa_idxs, inter_types, inter_mask, label, *_ = batch
            
            # 设备转移
            prot = prot.to(args.device)
            drug = drug.to(args.device)
            prot_mask = prot_mask.to(args.device)
            drug_mask = drug_mask.to(args.device)
            aa_types = aa_types.to(args.device)
            aa_idxs = aa_idxs.to(args.device)
            inter_types = inter_types.to(args.device)
            inter_mask = inter_mask.to(args.device)
            label = label.to(args.device)

            # 前向传播
            outputs = model(prot, drug, prot_mask, drug_mask, ...)  # 根据实际参数调整
            
            # 计算损失
            losses = evaluate_loss(
                model_output=outputs,
                targets=(aa_types, aa_idxs, inter_types, label),
                inter_mask=inter_mask,
                args=args
            )
            
            # 累加指标
            metric_log['total'] += losses['total_loss']
            metric_log['aa_types'] += losses['aa_types_loss']
            metric_log['aa_idxs'] += losses['aa_idxs_loss']
            metric_log['inter_types'] += losses['inter_types_loss']
            metric_log['affinity'] += losses['affinity_loss']
    
    # 计算平均指标
    num_batches = len(test_loader)
    return {k: v/num_batches for k, v in metric_log.items()}

