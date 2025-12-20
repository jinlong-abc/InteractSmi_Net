import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.model import FusionCPI
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

"""
使用方法:
python -m model.main > train_log.txt 2>&1
或者
python main.py --resume_from_checkpoint path/to/checkpoint.pt
"""


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


def setup_model_and_optimizer(atom_dict_len, params, device):
    """设置模型和优化器"""
    model = FusionCPI(atom_dict_len, params)
    model.to(device)
    
    criterion = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 打印模型结构
    print(model)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, criterion, optimizer, scheduler


def train_model(model, criterion, optimizer, scheduler, train_data, dev_data, test_data, 
                device, params, uniprotid_prot_embed_dict, start_epoch=0, start_step=0):
    """训练模型"""
    print('Starting training...')
    
    # 初始化训练状态
    global_step = start_step
    best_dev_rmse = float('inf')
    best_test_rmse = float('inf')
    idx = np.arange(len(train_data[0]))
    
    # 创建保存目录
    os.makedirs(params.checkpoint_path, exist_ok=True)
    best_model_dir = os.path.join(params.checkpoint_path, params.name, 'best_models')
    os.makedirs(best_model_dir, exist_ok=True)
    
    for epoch in range(start_epoch, params.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{params.num_epochs} ===")
        
        # 训练一个epoch
        global_step = train_one_epoch(
            global_step, model, criterion, optimizer, idx, 
            train_data, device, params, uniprotid_prot_embed_dict
        )
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # 评估模型
        dev_metrics, test_metrics = evaluate_model(
            model, dev_data, test_data, params.batch_size, device, uniprotid_prot_embed_dict
        )
        rmse_dev, pearson_dev, spearman_dev = dev_metrics
        rmse_test, pearson_test, spearman_test = test_metrics
        
        # 保存检查点
        if (epoch + 1) % params.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step, params.checkpoint_path, params.name)
        
        # 保存最佳模型
        best_dev_rmse = save_best_model(
            model, optimizer, epoch + 1, global_step, rmse_dev, best_dev_rmse,
            os.path.join(best_model_dir, 'best_dev_model.pt'), "Dev_RMSE"
        )
        
        best_test_rmse = save_best_model(
            model, optimizer, epoch + 1, global_step, rmse_test, best_test_rmse,
            os.path.join(best_model_dir, 'best_test_model.pt'), "Test_RMSE"
        )
        
        print(f"Best Dev RMSE so far: {best_dev_rmse:.4f}")
        print(f"Best Test RMSE so far: {best_test_rmse:.4f}")
        
        # 早停检查（可选）
        if hasattr(params, 'early_stopping') and params.early_stopping:
            if hasattr(params, 'patience') and epoch >= params.patience:
                # 实现早停逻辑
                pass
    
    return best_dev_rmse, best_test_rmse


def main():
    # 获取配置参数
    params = get_config()
    print('Parameters:', params)
    
    try:
        # 设置环境
        device = setup_environment(params)
        
        # 加载数据
        train_data, dev_data, test_data, atom_dict_len = prepare_datasets(params)
        
        # 加载蛋白质编码
        uniprotid_prot_embed_dict = load_protein_embeddings(params)
        
        # 设置模型和优化器
        model, criterion, optimizer, scheduler = setup_model_and_optimizer(
            atom_dict_len,  params, device
        )
        
        # 检查是否需要从检查点恢复
        start_epoch = 0
        start_step = 0
        if hasattr(params, 'resume_from_checkpoint') and params.resume_from_checkpoint:
            if os.path.exists(params.resume_from_checkpoint):
                model, optimizer, start_step = load_checkpoint(
                    params.resume_from_checkpoint, model, optimizer
                )
                start_epoch = start_step  # 简化假设每个epoch一个step
                print(f"Resumed training from epoch {start_epoch}")
            else:
                print(f"Checkpoint file not found: {params.resume_from_checkpoint}")
        
        # 训练模型
        best_dev_rmse, best_test_rmse = train_model(
            model, criterion, optimizer, scheduler, train_data, dev_data, test_data,
            device, params, uniprotid_prot_embed_dict, start_epoch, start_step
        )
        
        print('\n=== Training Completed ===')
        print(f'Best Dev RMSE: {best_dev_rmse:.4f}')
        print(f'Best Test RMSE: {best_test_rmse:.4f}')
        
        # 最终评估
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