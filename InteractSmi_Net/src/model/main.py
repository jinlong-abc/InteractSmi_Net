import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.model import FusionCPI
from model.decoder_utils import *
from data.data_utils import *
from data.data_processor import HDF5DataProcessor  
from config import get_config
from model.train_utils import (
    train_one_epoch, 
    save_checkpoint, 
    load_checkpoint, 
    test, 
    evaluate_model, 
    save_best_model
)


def setup_environment(params):
    """设置运行环境"""
    if params.mode == 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            print("CUDA is not available! Switching to CPU.")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    print(f'The code runs on: {device}')
    return device


def setup_model_and_optimizer(atom_dict_len, params, device, vocab=None,
                            multitask=False, use_adaptive_loss=False):
    """
    model
    criterion
    optimizer
    scheduler
    """
    model = FusionCPI(atom_dict_len, params, vocab)
    model.to(device)
    
    if multitask:
        if use_adaptive_loss:
            criterion = AdaptiveFusionCPILoss(
                initial_affinity_weight=params.affinity_weight,
                initial_sequence_weight=params.sequence_weight,
                adapt_strategy='loss_magnitude',
                ignore_index=-100
            )
            print("使用自适应联合损失函数")
        else:
            criterion = FusionCPILoss(
                affinity_weight=params.affinity_weight,
                sequence_weight=params.sequence_weight,
                ignore_index=-100
            )
            print(f"使用联合损失函数 - 亲和力权重: {params.affinity_weight}, 序列权重: {params.sequence_weight}")
    else:
        criterion = F.mse_loss
        print("使用传统MSE损失函数（仅亲和力预测）")
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, criterion, optimizer, scheduler


def train_model(model, criterion, optimizer, scheduler, train_data, dev_data, test_data, 
                device, params, uniprotid_prot_embed_dict, start_epoch=0, start_step=0, 
                vocab=None, multitask=False):
    print('Starting training...')

    global_step = start_step
    idx = np.arange(len(train_data[0]))

    if multitask:
        best_dev = {
            "loss": float("inf"),
            "affinity_loss": float("inf"),
            "sequence_loss": float("inf"),
            "rmse": float("inf")
        }
        best_test = {
            "loss": float("inf"),
            "affinity_loss": float("inf"),
            "sequence_loss": float("inf"),
            "rmse": float("inf")
        }
    else:
        best_dev = {
            "loss": float("inf"),
            "affinity_loss": None,
            "sequence_loss": None,
            "rmse": float("inf")
        }
        best_test = {
            "loss": float("inf"),
            "affinity_loss": None,
            "sequence_loss": None,
            "rmse": float("inf")
        }

    os.makedirs(params.checkpoint_path, exist_ok=True)
    dev_best_model_dir = os.path.join(params.checkpoint_path, params.name, 'dev_best_models')
    test_best_model_dir = os.path.join(params.checkpoint_path, params.name, 'test_best_models')
    os.makedirs(dev_best_model_dir, exist_ok=True)
    os.makedirs(test_best_model_dir, exist_ok=True)

    # 开始训练
    for epoch in range(start_epoch, params.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{params.num_epochs} ===")
        global_step = train_one_epoch(
            global_step, model, criterion, optimizer, idx, train_data, device, params, 
            uniprotid_prot_embed_dict, vocab=vocab, multitask=params.multitask
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        dev_metrics, test_metrics = evaluate_model(
            model, dev_data, test_data, params.batch_size, device, 
            criterion, uniprotid_prot_embed_dict, vocab=vocab, params=params
        )
        rmse_dev, pearson_dev, spearman_dev, avg_loss_dev, avg_affinity_loss_dev, avg_sequence_loss_dev = dev_metrics
        rmse_test, pearson_test, spearman_test, avg_loss_test, avg_affinity_loss_test, avg_sequence_loss_test = test_metrics

        if (epoch + 1) % params.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, params.checkpoint_path, params.name)

        metrics_dev = {
            "loss": avg_loss_dev,
            "affinity_loss": avg_affinity_loss_dev if multitask else None,
            "sequence_loss": avg_sequence_loss_dev if multitask else None,
            "rmse": rmse_dev
        }
        metrics_test = {
            "loss": avg_loss_test,
            "affinity_loss": avg_affinity_loss_test if multitask else None,
            "sequence_loss": avg_sequence_loss_test if multitask else None,
            "rmse": rmse_test
        }

        best_dev = save_best_model(
            model, optimizer, epoch + 1, global_step,
            metrics_dev, best_dev,
            save_dir=dev_best_model_dir,
            multitask=params.multitask
        )
        best_test = save_best_model(
            model, optimizer, epoch + 1, global_step,
            metrics_test, best_test,
            save_dir=test_best_model_dir,
            multitask=params.multitask
        )

        print(f"Best Dev RMSE so far: {best_dev['rmse']:.4f}")
        print(f"Best Test RMSE so far: {best_test['rmse']:.4f}")

        if hasattr(params, 'early_stopping') and params.early_stopping:
            if hasattr(params, 'patience') and epoch >= params.patience:
                # TODO: 实现早停逻辑
                pass

    return best_dev, best_test


def main():
    params = get_config()
    print('Parameters:', params)
    
    try:
        device = setup_environment(params)
        
        train_data, dev_data, test_data, atom_dict_len = prepare_datasets(params)
        
        uniprotid_prot_embed_dict = load_protein_embeddings(params)
        
        with open(params.vocab, "rb") as f:
            vocab = pickle.load(f)

        model, criterion, optimizer, scheduler = setup_model_and_optimizer(
            atom_dict_len,  params, device, vocab, multitask=params.multitask, use_adaptive_loss=False
        )
        
        start_epoch = 0
        start_step = 0
        if hasattr(params, 'resume_from_checkpoint') and params.resume_from_checkpoint:
            if os.path.exists(params.resume_from_checkpoint):
                model, optimizer, start_step = load_checkpoint(
                    params.resume_from_checkpoint, model, optimizer
                )
                start_epoch = start_step
                print(f"Resumed training from epoch {start_epoch}")
            else:
                print(f"Checkpoint file not found: {params.resume_from_checkpoint}")
        
        best_dev_rmse, best_test_rmse = train_model(
            model, criterion, optimizer, scheduler, train_data, dev_data, test_data, device, params,
            uniprotid_prot_embed_dict, start_epoch, start_step, vocab=vocab, multitask=params.multitask
        )
        
        print('\n=== Training Completed ===')
        print(f'Best Dev RMSE: {best_dev_rmse:.4f}')
        print(f'Best Test RMSE: {best_test_rmse:.4f}')
        
        print('\n=== Final Evaluation ===')
        best_model_path = os.path.join(params.checkpoint_path, 'best_models', 'best_dev_model.pt')
        if os.path.exists(best_model_path):
            print("Loading best dev model for final evaluation...")
            load_checkpoint(best_model_path, model)
            dev_metrics, test_metrics = evaluate_model(
                model, dev_data, test_data, params.batch_size, device, uniprotid_prot_embed_dict
            )
            print(f'Final Test Results - RMSE: {test_metrics[0]:.4f}, Pearson: {test_metrics[1]:.4f}, Spearman: {test_metrics[2]:.4f}')
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)