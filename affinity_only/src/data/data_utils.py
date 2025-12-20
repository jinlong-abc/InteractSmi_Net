import numpy as np
import torch
from torch.autograd import Variable
from math import sqrt
from scipy import stats
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import os
import pickle
from data.data_processor import HDF5DataProcessor
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

def prepare_datasets(params):
    """
    加载数据集，同时支持第一次处理保存全局字典，之后直接读取
    """
    # if params is None or not hasattr(params, 'dict_prefix'):
    #     data_dir = params.dict_prefix
    # if params is not None and hasattr(params, 'dict_prefix'):
    #     data_dir = params.dict_prefix
    # else:
    dataset = params.dataset
    data_dir = os.path.join('../datasets', dataset)
    bond_dict_path = os.path.join(data_dir, 'bond_dict.pkl')
    atom_dict_path = os.path.join(data_dir, 'atom_dict.pkl')  # 保存字典的路径
    fingerprint_dict_path = os.path.join(data_dir, 'fingerprint_dict.pkl')

    # 使用一个 processor 实例，保证全局词表一致
    processor = HDF5DataProcessor(args=params)
    if not os.path.isdir(data_dir) or not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Data directory {data_dir} not found or atom_dict missing. Processing data...")

        train_csv_path = params.train_csv_data_path
        train_output_path = params.train_dataset_path
        test_csv_path = params.test_csv_data_path
        test_output_path = params.test_dataset_path

        # 处理 train 和 test，保证字典累加
        processor.process_csv_to_hdf5(
            csv_path=train_csv_path,
            output_path=train_output_path,
            radius=params.radius,
            chunk_size=10000
        )
        processor.process_csv_to_hdf5(
            csv_path=test_csv_path,
            output_path=test_output_path,
            radius=params.radius,
            chunk_size=10000
        )

        # 保存全局字典到文件
        # 保存 atom_dict
        with open(fingerprint_dict_path, 'wb') as f:
            pickle.dump(dict(processor.graph_builder.fingerprint_dict), f)
        with open(atom_dict_path, 'wb') as f2:
            pickle.dump(dict(processor.graph_builder.atom_dict), f2)
        with open(bond_dict_path, 'wb') as f3:
            pickle.dump(dict(processor.graph_builder.bond_dict), f3)

        fingerprint_dict_len = len(processor.graph_builder.fingerprint_dict)
        print(f"First-time processing done. Total atom fingerprint types (train+test): {fingerprint_dict_len}")

    else:
        print(f"Loading existing atom_dict from {fingerprint_dict_path}...")
        with open(fingerprint_dict_path, 'rb') as f:
            fingerprint_dict = pickle.load(f)
        # processor.graph_builder.fingerprint_dict = defaultdict(lambda: len(fingerprint_dict), fingerprint_dict)
        processor.graph_builder.fingerprint_dict = fingerprint_dict

        fingerprint_dict_len = len(processor.graph_builder.fingerprint_dict)
        print(f"Loaded atom_dict. Total atom fingerprint types: {fingerprint_dict_len}")

    # 直接加载数据
    train_data = processor.load_dataset(params.train_dataset_path)
    test_data = processor.load_dataset(params.test_dataset_path)

    # 划分 dev 集
    train_data, dev_data = split_data(data_list=train_data, dev_ratio=0.1, seed=params.seed)

    print(f"Train samples: {len(train_data[0])}")
    print(f"Dev samples: {len(dev_data[0])}")
    print(f"Test samples: {len(test_data[0])}")
    print(f"Atom vocabulary size: {fingerprint_dict_len}")

    return train_data, dev_data, test_data, fingerprint_dict_len


def load_protein_embeddings(params):
    """加载蛋白质编码"""
    print('Loading protein embeddings...')
    if not os.path.exists(params.prot_esmc_embed):
        raise FileNotFoundError(f"Protein embedding file not found: {params.prot_esmc_embed}")
    
    with open(params.prot_esmc_embed, "rb") as f:
        uniprotid_prot_embed_dict = pickle.load(f)
    
    print(f"Loaded embeddings for {len(uniprotid_prot_embed_dict)} proteins")
    return uniprotid_prot_embed_dict


def split_data(data_list: List[np.ndarray], dev_ratio: float = 0.1, seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    np.random.seed(seed)
    
    # 获取样本数量（使用第一个数组的长度）
    n_samples = len(data_list[0])
    
    # 验证所有数组长度是否一致
    for i, data in enumerate(data_list):
        if len(data) != n_samples:
            raise ValueError(f"数据列表第{i}个元素长度不一致: {len(data)} != {n_samples}")
    
    # 生成随机索引
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 计算分割点
    n_dev = int(n_samples * dev_ratio)
    dev_idx = indices[:n_dev]
    train_idx = indices[n_dev:]
    
    print(f"数据分割完成:")
    print(f"  总样本数: {n_samples}")
    print(f"  训练集: {len(train_idx)} 样本 ({(1-dev_ratio)*100:.1f}%)")
    print(f"  验证集: {len(dev_idx)} 样本 ({dev_ratio*100:.1f}%)")
    
    # 按索引分割每个数组
    train_data = []
    dev_data = []
    
    for data in data_list:
        if isinstance(data, np.ndarray):
            train_subset = data[train_idx]
            dev_subset = data[dev_idx]
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
        
        train_data.append(train_subset)
        dev_data.append(dev_subset)
    
    return train_data, dev_data

def batch_pad(arr):
    """批次填充函数"""
    N = max([a.shape[0] for a in arr])
    if arr[0].ndim == 1:
        new_arr = np.zeros((len(arr), N))
        new_arr_mask = np.zeros((len(arr), N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n] = a + 1
            new_arr_mask[i, :n] = 1
        return new_arr, new_arr_mask

    elif arr[0].ndim == 2:
        new_arr = np.zeros((len(arr), N, N))
        new_arr_mask = np.zeros((len(arr), N, N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n, :n] = a
            new_arr_mask[i, :n, :n] = 1
        return new_arr, new_arr_mask

def batch2tensor(batch_data, device):
    """将批次数据转换为tensor"""
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adjacencies_pad, _ = batch_pad(batch_data[1])

    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
    label = torch.FloatTensor(batch_data[3]).to(device) 
    id_list = batch_data[4]
    uniprot_list = batch_data[5]

    return atoms_pad, atoms_mask, adjacencies_pad, label, id_list, uniprot_list

def load_hdf5_data(hdf5_path):
    """从HDF5文件加载数据"""
    processor = HDF5DataProcessor()
    data = processor.load_dataset(hdf5_path)
    
    # 转换为原始格式
    compounds = data['compounds']
    adjacencies = data['adjacencies']
    # 如果没有指纹数据，创建空列表
    fingerprints = data.get('fingerprints', [None] * len(compounds))
    proteins = data['proteins']
    interactions = data['interactions']
    id_list = data['ID_list']
    uniprot_list = data['Uniprot_list']
    
    data_pack = [compounds, adjacencies, fingerprints, proteins, interactions, id_list, uniprot_list]
    return data_pack


def regression_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(rmse, 6), round(pearson, 6), round(spearman, 6)

def classification_scores(label, pred_score, pred_label):
    label = label.reshape(-1)
    pred_score = pred_score.reshape(-1)
    pred_label = pred_label.reshape(-1)
    auc = roc_auc_score(label, pred_score)
    acc = accuracy_score(label, pred_label)
    precision, recall, _ = precision_recall_curve(label, pred_label)
    aupr = metrics.auc(recall, precision)
    return round(auc, 6), round(acc, 6), round(aupr, 6)

def concordance_index_fast(y_true, y_pred):
    """
    Efficient Concordance Index (CI) calculation in O(n log n).
    Handles ties approximately (prediction ties = 0.5 credit).
    """
    n = len(y_true)
    # 排序：按真实值递增
    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    # 压缩预测值为整数排名 (1..m)，便于用树状数组统计
    unique_preds, inv = np.unique(y_pred_sorted, return_inverse=True)
    ranks = inv + 1  # 从 1 开始

    # Fenwick Tree
    size = len(unique_preds) + 2
    bit_counts = np.zeros(size, dtype=np.int64)

    def bit_update(bit, idx, val):
        while idx < len(bit):
            bit[idx] += val
            idx += idx & -idx

    def bit_query(bit, idx):
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    concordant = 0
    permissible = 0

    # 遍历所有样本，统计与之前样本的比较
    for i in range(n):
        r = ranks[i]
        # 已见过的样本数
        seen = bit_query(bit_counts, size - 1)
        # 比当前预测小的数
        less = bit_query(bit_counts, r - 1)
        # 比当前预测大的数
        greater = seen - bit_query(bit_counts, r)

        # 当前样本与所有真实值更小的样本配对
        if i > 0:
            permissible += seen
            concordant += less + 0.5 * (seen - less - greater)

        # 更新 BIT
        bit_update(bit_counts, r, 1)

    return concordant / permissible if permissible > 0 else np.nan


def regression_scores_full(label, pred):
    """
    Calculate regression evaluation metrics:
    RMSE, MAE, Pearson r, Spearman ρ, Modified Rm^2, Concordance Index (CI).
    """
    label = label.reshape(-1)
    pred = pred.reshape(-1)

    # RMSE
    rmse = np.sqrt(((label - pred) ** 2).mean())

    # MAE
    mae = np.mean(np.abs(label - pred))

    # Pearson correlation (r)
    r = np.corrcoef(label, pred)[0, 1]

    # Spearman correlation (ρ)
    rho = stats.spearmanr(label, pred)[0]

    # R²
    R2 = r ** 2

    # R0² (force regression through origin: y = kx)
    slope1 = np.dot(label, pred) / np.dot(label, label)
    R0_sq_1 = (np.corrcoef(pred, slope1 * label)[0, 1]) ** 2

    # R0'² (reverse: force regression pred vs label)
    slope2 = np.dot(pred, label) / np.dot(pred, pred)
    R0_sq_2 = (np.corrcoef(label, slope2 * pred)[0, 1]) ** 2

    # Modified Rm² (two-way average)
    from math import sqrt
    Rm2_1 = R2 * (1 - sqrt(abs(R2 - R0_sq_1)))
    Rm2_2 = R2 * (1 - sqrt(abs(R2 - R0_sq_2)))
    Rm2 = (Rm2_1 + Rm2_2) / 2

    # CI (fast version)
    ci = concordance_index_fast(label, pred)

    return {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "r": round(r, 6),
        "rho": round(rho, 6),
        "Rm2": round(Rm2, 6),
        "CI": round(ci, 6)
    }


# def regression_scores_full(label, pred):
#     """
#     Calculate regression evaluation metrics:
#     RMSE, MAE, Pearson r, Spearman ρ, Modified Rm^2, Concordance Index (CI).
#     大量数据计算Cl会很慢
#     """
#     label = label.reshape(-1)
#     pred = pred.reshape(-1)

#     # RMSE
#     rmse = np.sqrt(((label - pred) ** 2).mean())

#     # MAE
#     mae = np.mean(np.abs(label - pred))

#     # Pearson correlation (r)
#     r = np.corrcoef(label, pred)[0, 1]

#     # Spearman correlation (ρ)
#     rho = stats.spearmanr(label, pred)[0]

#     # R²
#     R2 = r ** 2

#     # R0² (force regression through origin: y = kx)
#     slope1 = np.dot(label, pred) / np.dot(label, label)
#     R0_sq_1 = (np.corrcoef(pred, slope1 * label)[0, 1]) ** 2

#     # R0'² (reverse: force regression pred vs label)
#     slope2 = np.dot(pred, label) / np.dot(pred, pred)
#     R0_sq_2 = (np.corrcoef(label, slope2 * pred)[0, 1]) ** 2

#     # Modified Rm² (two-way average)
#     Rm2_1 = R2 * (1 - sqrt(abs(R2 - R0_sq_1)))
#     Rm2_2 = R2 * (1 - sqrt(abs(R2 - R0_sq_2)))
#     Rm2 = (Rm2_1 + Rm2_2) / 2

#     # Concordance Index (CI)
#     concordant, permissible = 0, 0
#     n = len(label)
#     for i in range(n):
#         for j in range(i+1, n):
#             if label[i] != label[j]:
#                 permissible += 1
#                 if (pred[i] - pred[j]) * (label[i] - label[j]) > 0:
#                     concordant += 1
#                 elif (pred[i] - pred[j]) == 0:
#                     concordant += 0.5
#     ci = concordant / permissible if permissible > 0 else np.nan

#     return {
#         "RMSE": round(rmse, 6),
#         "MAE": round(mae, 6),
#         "r": round(r, 6),
#         "rho": round(rho, 6),
#         "Rm2": round(Rm2, 6),
#         "CI": round(ci, 6)
#     }

# 使用示例
if __name__ == "__main__":
    # 创建一个模拟的参数对象
    class Params:
        def __init__(self):
            self.dataset = 'test_dataset'
            self.train_csv_path = 'train.csv'
            self.test_csv_path = 'test.csv'
            self.batch_size = 32
    
    params = Params()
    
    # 加载数据集
    train_data, dev_data, test_data, atom_dict, amino_dict = prepare_datasets(params)
    
    print("数据加载完成！")