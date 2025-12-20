from test.Test_Config import get_config
import numpy as np
from math import sqrt
from scipy import stats
import re

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
    label = np.array(label).astype(float)
    pred = np.array(pred).astype(float)
    # label = label.reshape(-1)
    # pred = pred.reshape(-1)

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
#     """
#     label = np.array(label).reshape(-1).astype(float)
#     pred = np.array(pred).reshape(-1).astype(float)

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


if __name__ == '__main__':
    # 参数获取
    test_params = get_config()

    # 读取真实亲和力标签
    label_list = []
    with open(test_params.test_data, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            line = line.strip().split(',')
            label_list.append(float(line[3]))  # 第4列是label

    # 从日志文件提取预测值
    pred_list = []
    with open("/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/src/pdbbind2020_364_test.txt", "r") as f:  # 这里替换成你的日志文件路径
        for line in f:
            if "Predicted affinity:" in line:
                value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                # ===调试===
                # print(f"Extracted predicted value: {value}")
                pred_list.append(value)

    # 确认数量对齐
    print(f"Labels: {len(label_list)}, Predictions: {len(pred_list)}")
    # ===调试===
    # print(f"First 5 labels: {label_list[:5]}")
    # print(f"First 5 predictions: {pred_list[:5]}")
    # print(f"Last 5 labels: {label_list[-5:]}")
    # print(f"Last 5 predictions: {pred_list[-5:]}")
    # a = input()

    # 计算指标
    scores = regression_scores_full(label_list[:len(pred_list)], pred_list)
    print("Evaluation results:")
    for k, v in scores.items():
        print(f"{k}: {v}")

"""
 python -m test.Test_Evalue
"""

# Evaluation results:
# RMSE: 1.202159
# MAE: 0.867357
# r: 0.828849
# rho: 0.815876
# Rm2: 0.68699
# CI: 0.818867