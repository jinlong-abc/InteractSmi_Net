from data.graph_builder import MolecularGraphBuilder
import pickle
import numpy as np
from math import sqrt
from scipy import stats

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
    Rm2_1 = R2 * (1 - sqrt(abs(R2 - R0_sq_1)))
    Rm2_2 = R2 * (1 - sqrt(abs(R2 - R0_sq_2)))
    Rm2 = (Rm2_1 + Rm2_2) / 2

    # Concordance Index (CI)
    concordant, permissible = 0, 0
    n = len(label)
    for i in range(n):
        for j in range(i+1, n):
            if label[i] != label[j]:
                permissible += 1
                if (pred[i] - pred[j]) * (label[i] - label[j]) > 0:
                    concordant += 1
                elif (pred[i] - pred[j]) == 0:
                    concordant += 0.5
    ci = concordant / permissible if permissible > 0 else np.nan

    return {
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "r": round(r, 6),
        "rho": round(rho, 6),
        "Rm2": round(Rm2, 6),
        "CI": round(ci, 6)
    }




# # 按 batch2tensor 需要的顺序返回列表
# data_list = [
#     np.array(compounds, dtype=object),     # 0
#     np.array(adjacencies, dtype=object),   # 1
#     np.array(proteins, dtype=object),      # 2
#     np.array(interactions, dtype=float),   # 3
#     np.array(ID_list, dtype=object),       # 4
#     np.array(Uniprot_list, dtype=object),  # 5
#     np.array(smiles_list, dtype=object),   # 6
#     np.array(fusionsmi_list, dtype=object) # 7
# ]

# 构建compounds图
builder = MolecularGraphBuilder()
smiles_list = [
    "CCO",  
    "C1=CC=CC=C1",  
]
compounds, adjacencies = builder.batch_mol_to_graph(smiles_list)
# print(f"成功处理 {len(compounds)} 个分子")
# print("字典统计:", builder.get_statistics())
# builder.save_dictionaries("./dictionaries")

# 分子特征数目
atom_dict_path = '/home/wjl/data/DTI_prj/Arch_Lab/DecoderOptionalArch/datasets/Pdbbind/atom_dict.pkl'
with open(atom_dict_path, 'rb') as f:
    atom_dict = pickle.load(f)
atom_dict_len = len(atom_dict)

# 

