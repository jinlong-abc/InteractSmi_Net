import math
import sys
import torch
import numpy as np
from data.data_utils import *
import os
import glob


def save_checkpoint(model, optimizer, epoch, step, save_dir, name, max_keep=3):
    """保存模型检查点"""
    save_dir_full = os.path.join(save_dir, name)
    os.makedirs(save_dir_full, exist_ok=True)
    save_path = os.path.join(save_dir_full, f"checkpoint_epoch{epoch}_step{step}.pt")

    # 处理分布式训练的情况
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'global_step': step,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict()
    }, save_path)
    print(f"[✔] Checkpoint saved at: checkpoint_epoch{epoch}_step{step}.pt")

    # 删除旧的 checkpoint，保留最新 max_keep 个
    checkpoints = sorted(glob.glob(os.path.join(save_dir_full, "checkpoint_epoch*_step*.pt")),
                         key=os.path.getmtime, reverse=True)
    if len(checkpoints) > max_keep:
        for ckpt_path in checkpoints[max_keep:]:
            os.remove(ckpt_path)
            print(f"[🗑] Removed old checkpoint: {os.path.basename(ckpt_path)}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None, 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    print(f"[✔] Loaded checkpoint from epoch {epoch}, step {global_step}")
    return model, optimizer, global_step


def process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device):
    """处理蛋白质编码数据"""
    batch_prot_embed = []
    batch_prot_mask = []
    
    # 将numpy数组转换为列表
    if isinstance(uniprot_list, np.ndarray):
        uniprot_list = uniprot_list.tolist()
    
    for uid in uniprot_list:
        # 处理不同类型的uid
        if isinstance(uid, torch.Tensor):
            uid = uid.item() if uid.ndim == 0 else str(uid)
        elif isinstance(uid, bytes):
            uid = uid.decode("utf-8")
        elif isinstance(uid, (int, float)):
            uid = str(uid)
        
        # 检查uid是否在字典中
        if uid not in uniprotid_prot_embed_dict:
            raise ValueError(f"UniProt ID {uid} not found in embedding dictionary")
        
        # 从字典中获取embed和mask
        batch_prot_embed.append(uniprotid_prot_embed_dict[uid]["embed"])
        batch_prot_mask.append(uniprotid_prot_embed_dict[uid]["mask"])
    
    # 合并成batch维度 [batch_size, seq_len, emb_dim]
    batch_prot_embed = torch.stack(batch_prot_embed, dim=0).to(device)
    batch_prot_mask = torch.stack(batch_prot_mask, dim=0).to(device)
    
    return batch_prot_embed, batch_prot_mask


def train_one_epoch(global_step, model, criterion, optimizer, idx, train_data, device, params, uniprotid_prot_embed_dict=None):
    """训练一个epoch"""
    np.random.shuffle(idx)
    model.train()
    
    predictions = []
    labels = []
    total_loss = 0.0
    batch_size = params.batch_size
    num_batches = math.ceil(len(train_data[0]) / batch_size)
    
    for i in range(num_batches):
        # 准备批次数据
        batch_indices = idx[i * batch_size: (i + 1) * batch_size]
        batch_data = [train_data[di][batch_indices] for di in range(len(train_data))]
        atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list = batch2tensor(batch_data, device)

        # 处理蛋白质编码
        prot_embed, prot_mask = process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device)

        # 前向传播
        pred = model(atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)
        label = label.view(-1, 1)  # [B] -> [B,1]
        loss = criterion(pred.float(), label.float())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        
        # 记录结果
        predictions.extend(pred.cpu().detach().numpy().reshape(-1).tolist())
        labels.extend(label.cpu().numpy().reshape(-1).tolist())
        total_loss += loss.item()

        # 打印进度
        if params.verbose and (i % max(num_batches // 1, 1) == 0 or i == num_batches - 1):
            avg_loss = total_loss / (i + 1)
            progress = (i + 1) / num_batches * 100
            sys.stdout.write(f'\repoch: {global_step}, batch: {i+1}/{num_batches} ({progress:.1f}%), avg_loss: {avg_loss:.6f}')
            sys.stdout.flush()
        
        # except Exception as e:
        #     print(f"\nError in batch {i}: {str(e)}")
        #     continue

    print()  # 换行
    
    # 计算训练指标
    predictions = np.array(predictions)
    labels = np.array(labels)
    metrics = regression_scores_full(labels, predictions)
    # rmse_train, pearson_train, spearman_train = regression_scores(labels, predictions)
    avg_loss = total_loss / num_batches

    print(f'Train - Loss: {avg_loss:.6f}, RMSE: {metrics["RMSE"]:.4f}, Pearson: {metrics["r"]:.4f}, Spearman: {metrics["rho"]:.4f} \
          MAE: {metrics["MAE"]:.4f}, Rm2: {metrics["Rm2"]:.4f}, CI: {metrics["CI"]:.4f}')

    return global_step


def test(model, data_test, batch_size, device, uniprotid_prot_embed_dict=None):
    """测试模型"""
    model.eval()
    predictions = []
    labels = []
    num_batches = math.ceil(len(data_test[0]) / batch_size)
    
    with torch.no_grad():
        for i in range(num_batches):
            try:
                batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_test))]

                atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list = batch2tensor(batch_data, device)
                # 处理蛋白质编码
                prot_embed, prot_mask = process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device)
                # 前向传播
                pred = model(atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)
                
                predictions.extend(pred.cpu().detach().numpy().reshape(-1).tolist())
                labels.extend(label.cpu().numpy().reshape(-1).tolist())
                
            except Exception as e:
                print(f"Error in test batch {i}: {str(e)}")
                continue

    # 计算测试指标
    predictions = np.array(predictions)
    labels = np.array(labels)
    # rmse_value, pearson_value, spearman_value = regression_scores(labels, predictions)
    metrics = regression_scores_full(labels, predictions)
    rmse_value = metrics["RMSE"]
    mae_value = metrics["MAE"]
    pearson_value = metrics["r"]
    spearman_value = metrics["rho"]
    rm2_value = metrics["Rm2"]
    ci_value = metrics["CI"]

    return rmse_value, mae_value, pearson_value, spearman_value, rm2_value, ci_value


def evaluate_model(model, dev_data, test_data, batch_size, device, uniprotid_prot_embed_dict):
    """评估模型在验证集和测试集上的性能"""
    # 验证集评估rmse_value, mae_value, pearson_value, spearman_value, rm2_value, ci_value
    rmse_dev, mae_dev, pearson_dev, spearman_dev, rm2_dev, ci_dev\
          = test(model, dev_data, batch_size, device, uniprotid_prot_embed_dict)
    print(f'Dev - RMSE: {rmse_dev:.4f}, MAE: {mae_dev:.4f}, Pearson: {pearson_dev:.4f}, \
          Spearman: {spearman_dev:.4f}, Rm2: {rm2_dev:.4f}, CI: {ci_dev:.4f}')

    # 测试集评估rmse_value, mae_value, pearson_value, spearman_value, rm2_value, ci_value
    rmse_test, mae_test, pearson_test, spearman_test, rm2_test, ci_test\
          = test(model, test_data, batch_size, device, uniprotid_prot_embed_dict)
    print(f'Test - RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, Pearson: {pearson_test:.4f}, \
          Spearman: {spearman_test:.4f}, Rm2: {rm2_test:.4f}, CI: {ci_test:.4f}')
    
    return (rmse_dev, pearson_dev, spearman_dev), (rmse_test, pearson_test, spearman_test)


def save_best_model(model, optimizer, epoch, step, metric_value, best_value, save_path, metric_name="RMSE", lower_is_better=True):
    """保存最佳模型"""
    is_best = (metric_value < best_value) if lower_is_better else (metric_value > best_value)
    
    if is_best:
        torch.save({
            'epoch': epoch,
            'global_step': step,
            'model_state': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            f'best_{metric_name.lower()}': metric_value
        }, save_path)
        print(f"[✔] Best model saved with {metric_name}: {metric_value:.4f}")
        return metric_value
    
    return best_value