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
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'global_step': step,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict()
    }, save_path)
    print(f"[✔] Checkpoint saved at: checkpoint_epoch{epoch}_step{step}.pt")

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
    
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint['model_state'])
    
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
    
    if isinstance(uniprot_list, np.ndarray):
        uniprot_list = uniprot_list.tolist()
    
    for uid in uniprot_list:
        if isinstance(uid, torch.Tensor):
            uid = uid.item() if uid.ndim == 0 else str(uid)
        elif isinstance(uid, bytes):
            uid = uid.decode("utf-8")
        elif isinstance(uid, (int, float)):
            uid = str(uid)
        
        if uid not in uniprotid_prot_embed_dict:
            raise ValueError(f"UniProt ID {uid} not found in embedding dictionary")
        
        batch_prot_embed.append(uniprotid_prot_embed_dict[uid]["embed"])
        batch_prot_mask.append(uniprotid_prot_embed_dict[uid]["mask"])
    
    batch_prot_embed = torch.stack(batch_prot_embed, dim=0).to(device)
    batch_prot_mask = torch.stack(batch_prot_mask, dim=0).to(device)
    
    return batch_prot_embed, batch_prot_mask

def train_one_epoch(global_step, model, criterion, optimizer, idx, train_data, device, 
                    params, uniprotid_prot_embed_dict=None, vocab=None, multitask=False):
    np.random.shuffle(idx)
    model.train()
    
    predictions, labels = [], []
    total_loss = 0.0
    total_affinity_loss = 0.0
    total_sequence_loss = 0.0
    
    batch_size = params.batch_size
    num_batches = math.ceil(len(train_data[0]) / batch_size)
    
    for i in range(num_batches):
        batch_indices = idx[i * batch_size: (i + 1) * batch_size]
        batch_data = [train_data[di][batch_indices] for di in range(len(train_data))]
        if multitask:
            atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list, Smiles, \
            Smiles_mask, FusionSmi, FusionSmi_mask, Sequence, Sequence_mask = batch2tensor(
                batch_data, device, vocab=vocab, params=params
            )
            prot_embed, prot_mask = process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device)
            pred, sequence_logits, fusionsmi_labels = model(
                atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask,
                Smiles, Smiles_mask, FusionSmi, FusionSmi_mask, Sequence, Sequence_mask
            )
            
            loss, affinity_loss, sequence_loss = criterion(
                pred, sequence_logits, fusionsmi_labels, target_affinity=label
            )
            
        else:
            atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list, Smiles, \
            Smiles_mask, FusionSmi, FusionSmi_mask, Sequence, Sequence_mask = batch2tensor(
                batch_data, device, vocab=vocab, params=params
            )
            prot_embed, prot_mask = process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device)
            
            pred = model(atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)
            label = label.view(-1, 1)  # [B] -> [B,1]
            loss = criterion(pred.float(), label.float())
            affinity_loss, sequence_loss = loss, None
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        
        predictions.extend(pred.cpu().detach().numpy().reshape(-1).tolist())
        labels.extend(label.cpu().numpy().reshape(-1).tolist())
        total_loss += loss.item()
        total_affinity_loss += affinity_loss.item() if affinity_loss is not None else 0
        total_sequence_loss += sequence_loss.item() if sequence_loss is not None else 0
        
        if params.verbose and (i % max(num_batches // 1, 1) == 0 or i == num_batches - 1):
            avg_loss = total_loss / (i + 1)
            avg_affinity_loss = total_affinity_loss / (i + 1)
            avg_sequence_loss = total_sequence_loss / (i + 1) if multitask else 0
            progress = (i + 1) / num_batches * 100
            if multitask:
                sys.stdout.write(f'\repoch: {global_step}, batch: {i+1}/{num_batches} ({progress:.1f}%), '
                                 f'total_loss: {avg_loss:.6f}, affinity_loss: {avg_affinity_loss:.6f}, '
                                 f'sequence_loss: {avg_sequence_loss:.6f}')
            else:
                sys.stdout.write(f'\repoch: {global_step}, batch: {i+1}/{num_batches} ({progress:.1f}%), '
                                 f'avg_loss: {avg_loss:.6f}')
            sys.stdout.flush()
    
    print()  

    predictions = np.array(predictions)
    labels = np.array(labels)
    rmse_train, pearson_train, spearman_train = regression_scores(labels, predictions)
    
    avg_loss = total_loss / num_batches
    avg_affinity_loss = total_affinity_loss / num_batches
    avg_sequence_loss = total_sequence_loss / num_batches if multitask else 0
    
    if multitask:
        print(f'Train Affinity - Loss: {avg_affinity_loss:.6f}, RMSE: {rmse_train:.4f}, '
              f'Pearson: {pearson_train:.4f}, Spearman: {spearman_train:.4f}')
        print(f'Train Sequence - Loss: {avg_sequence_loss:.6f}')
        print(f'Train Total Loss: {avg_loss:.6f}')
    else:
        print(f'Train - Loss: {avg_loss:.6f}, RMSE: {rmse_train:.4f}, '
              f'Pearson: {pearson_train:.4f}, Spearman: {spearman_train:.4f}')
    
    return global_step



def evaluate_model(model, dev_data, test_data, batch_size, device, 
                   criterion, uniprotid_prot_embed_dict, vocab=None, params=None):
    """评估模型在验证集和测试集上的性能（支持单任务 / 多任务）"""
    print("=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)

    # Dev
    print("\nDevelopment Set:")
    rmse_dev, mae_dev, pearson_dev, spearman_dev, rm2_dev, ci_dev, \
        avg_loss_dev, avg_aff_loss_dev, avg_seq_loss_dev = test(
        model, dev_data, batch_size, device, criterion,
        uniprotid_prot_embed_dict, vocab=vocab, params=params, multitask=params.multitask
    )
    if params.multitask:
        print(f'Affinity - Loss: {avg_aff_loss_dev:.6f}, RMSE: {rmse_dev:.4f}, Pearson: {pearson_dev:.4f},\
               Spearman: {spearman_dev:.4f}, RM2: {rm2_dev:.4f}, CI: {ci_dev:.4f}, MAE: {mae_dev:.4f}')
        print(f'Sequence - Loss: {avg_seq_loss_dev:.6f}')
    else:
        print(f'RMSE: {rmse_dev:.4f}, Pearson: {pearson_dev:.4f}, Spearman: {spearman_dev:.4f},\
               RM2: {rm2_dev:.4f}, CI: {ci_dev:.4f}, MAE: {mae_dev:.4f}')
    print(f'Total Loss: {avg_loss_dev:.6f}')

    # Test
    print("\nTest Set:")
    rmse_test, mae_test, pearson_test, spearman_test, rm2_test, ci_test, \
     avg_loss_test, avg_aff_loss_test, avg_seq_loss_test = test(
        model, test_data, batch_size, device, criterion,
        uniprotid_prot_embed_dict, vocab=vocab, params=params, multitask=params.multitask
    )
    if params.multitask:
        print(f'Affinity - Loss: {avg_aff_loss_test:.6f}, RMSE: {rmse_test:.4f}, Pearson: {pearson_test:.4f},\
               Spearman: {spearman_test:.4f}, RM2: {rm2_test:.4f}, CI: {ci_test:.4f}, MAE: {mae_test:.4f}')
        print(f'Sequence - Loss: {avg_seq_loss_test:.6f}')
    else:
        print(f'RMSE: {rmse_test:.4f}, Pearson: {pearson_test:.4f}, Spearman: {spearman_test:.4f},\
               RM2: {rm2_test:.4f}, CI: {ci_test:.4f}, MAE: {mae_test:.4f}')
    print(f'Total Loss: {avg_loss_test:.6f}')

    print("=" * 60)

    return (
        rmse_dev, pearson_dev, spearman_dev, avg_loss_dev, avg_aff_loss_dev, avg_seq_loss_dev
    ), (
        rmse_test, pearson_test, spearman_test, avg_loss_test, avg_aff_loss_test, avg_seq_loss_test
    )


def test(model, data_test, batch_size, device, criterion=None, 
         uniprotid_prot_embed_dict=None, vocab=None, params=None, multitask=False):
    """测试模型（支持单任务 / 多任务）"""
    model.eval()
    predictions, labels = [], []
    total_loss, total_affinity_loss, total_sequence_loss = 0.0, 0.0, 0.0

    num_batches = math.ceil(len(data_test[0]) / batch_size)

    with torch.no_grad():
        for i in range(num_batches):
            try:
                batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] 
                              for di in range(len(data_test))]

                if multitask:
                    atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list, Smiles_list, \
                    Smiles_mask, FusionSmi_list, FusionSmi_mask, Sequence_list, Sequence_mask = batch2tensor(
                        batch_data, device, vocab=vocab, params=params
                    )
                    prot_embed, prot_mask = process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device)

                    pred, sequence_logits, fusionsmi_labels = model(
                        atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask,
                        Smiles_list, Smiles_mask, FusionSmi_list, FusionSmi_mask, Sequence_list, Sequence_mask
                    )

                    loss, affinity_loss, sequence_loss = criterion(
                        pred, sequence_logits, fusionsmi_labels, target_affinity=label
                    )

                else:
                    atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list, Smiles_list, \
                    Smiles_mask, FusionSmi_list, FusionSmi_mask, Sequence_list, Sequence_mask = batch2tensor(
                        batch_data, device, vocab=vocab, params=params
                    )
                    prot_embed, prot_mask = process_protein_embeddings(uniprot_list, uniprotid_prot_embed_dict, device)

                    pred = model(atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)
                    label = label.view(-1, 1)
                    loss = criterion(pred.float(), label.float()) if criterion is not None else torch.tensor(0.0)
                    affinity_loss, sequence_loss = loss, None

                total_loss += loss.item()
                total_affinity_loss += affinity_loss.item() if affinity_loss is not None else 0
                total_sequence_loss += sequence_loss.item() if sequence_loss is not None else 0

                predictions.extend(pred.cpu().detach().numpy().reshape(-1).tolist())
                labels.extend(label.cpu().numpy().reshape(-1).tolist())

            except Exception as e:
                print(f"Error in test batch {i}: {str(e)}")
                continue

    predictions, labels = np.array(predictions), np.array(labels)
    metrics = regression_scores_full(labels, predictions)
    rmse_value = metrics["RMSE"]
    mae_value = metrics["MAE"]
    pearson_value = metrics["r"]
    spearman_value = metrics["rho"]
    rm2_value = metrics["Rm2"]
    ci_value = metrics["CI"]

    avg_loss = total_loss / num_batches
    avg_affinity_loss = total_affinity_loss / num_batches
    avg_sequence_loss = total_sequence_loss / num_batches if multitask else 0

    return rmse_value, mae_value, pearson_value, spearman_value, rm2_value, \
           ci_value, avg_loss, avg_affinity_loss, avg_sequence_loss


def save_best_model(model, optimizer, epoch, step,
                    metrics, best_values, save_dir, multitask=False):
    os.makedirs(save_dir, exist_ok=True)
    improved = {}

    lower_is_better = {
        "loss": True,
        "affinity_loss": True,
        "sequence_loss": True,
        "rmse": True
    }

    check_keys = ["loss", "rmse"]
    if multitask:
        check_keys.extend(["affinity_loss", "sequence_loss"])

    for key in check_keys:
        value = metrics.get(key, None)
        if value is None: 
            continue

        best_val = best_values.get(key, float("inf") if lower_is_better[key] else -float("inf"))
        if (lower_is_better[key] and value < best_val) or (not lower_is_better[key] and value > best_val):
            improved[key] = (best_val, value)
            best_values[key] = value

            save_path = os.path.join(save_dir, f"best_{key}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": step,
                "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_values": best_values,
                "improved_metric": key
            }, save_path)
            print(f"[✔] Best model (by {key}) saved to {save_path}: {best_val:.4f} -> {value:.4f}")

    return best_values
