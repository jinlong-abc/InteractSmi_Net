import torch
import torch
import os
import pickle
import math
from test.Test_GraphBuilder import MolecularGraphBuilder_TEST
from data.data_utils import *
from test.Test_Config import get_config
from model.model import FusionCPI
# from model.model_no_tishi import FusionCPI
from torch.nn import functional as F
from test.Test_Evalue import regression_scores_full

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


# ======================蛋白编码=================================================================
def load_protein_embeddings(params):
    """加载蛋白质编码"""
    print('Loading protein embeddings...')
    if not os.path.exists(params.prot_esmc_embed):
        raise FileNotFoundError(f"Protein embedding file not found: {params.prot_esmc_embed}")
    
    with open(params.prot_esmc_embed, "rb") as f:
        uniprotid_prot_embed_dict = pickle.load(f)
    
    print(f"Loaded embeddings for {len(uniprotid_prot_embed_dict)} proteins")
    return uniprotid_prot_embed_dict

# ====================smiles_seq_pad==============================================================
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


def batch2tensor_test(batch_data, device):
    """
    将批次数据转换为tensor
        np.array(compounds, dtype=object),     
        np.array(adjacencies, dtype=object),   
        np.array(prot_sep_list, dtype=object), 
        np.array(smiles_list, dtype=object),   
    """
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adjacencies_pad, _ = batch_pad(batch_data[1])

    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)

    return atoms_pad, atoms_mask, adjacencies_pad

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

def test_model(model, checkpoint_path, batch_data, device):
    """
    test_model(model, test_params.best_model, batch_data, device)
    batch_data = (atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)
    def inference(model, atoms, atoms_mask, adjacency, prot_embed, prot_mask, device):
    """
    # # 1. 加载模型参数
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint  # 说明就是纯 state_dict
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        # 2. 前向推理 (调用 inference 而不是 forward)
        inputs = [x.to(device) if torch.is_tensor(x) else x for x in batch_data]
        # print(f'inputs:{inputs}')

        prediction = model.inference(*inputs, device)
        # 3. 打印亲和力预测值
        print(f"Predicted affinity: {prediction.squeeze().cpu().numpy()}")
    return prediction


if __name__ == '__main__':
    # 参数获取
    test_params = get_config()

    builder = MolecularGraphBuilder_TEST(
        test_params.fingerprint_dict_path, 
        test_params.atom_dict_path, 
        test_params.bond_dict_path
    )
    # ID,UniProt,Assay Result,pInteraction,Normal SMILES,SMILES,FusionSmi,Mapped_SMILES,IntSeq,Sequence
    # normal_smiles_list = []
    # uniprotid_list = []
    # ID_list = []
    # label_list = []
    # with open(test_params.test_data, 'r') as f:
    #     next(f)  # 跳过标题行
    #     for line in f:
    #         line = line.strip().split(',')
    #         normal_smiles_list.append(line[4])
    #         uniprotid_list.append(line[1])
    #         label_list.append(line[3])
    #         ID_list.append(line[0])
    import csv
    normal_smiles_list = []
    uniprotid_list = []
    ID_list = []
    label_list = []
    # ID,UniProt,Ligand,Protein,Target_Chain,regression_label,Sequence
    with open(test_params.test_data, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for line in reader:
            ID_list.append(line[0])
            uniprotid_list.append(line[1])
            label_list.append(line[3])
            normal_smiles_list.append(line[4])

    # device
    device = setup_environment(test_params)

    # 构建小分子图（两个numpy的list） ==> pad
    compounds, adjacencies = builder.batch_mol_to_graph(normal_smiles_list, radius=1)

    # 加载蛋白质编码
    uniprotid_prot_embed_dict = load_protein_embeddings(test_params)

    # ===============================
    # 这里开始加入 batch_size 的循环
    # ===============================
    batch_size = test_params.batch_size
    num_batches = math.ceil(len(compounds) / batch_size)
    print(f"Total samples: {len(compounds)}, Batch size: {batch_size}, Num batches: {num_batches}")

    # 加载模型
    with open(test_params.fingerprint_dict_path, 'rb') as f:
        fingerprint_dict = pickle.load(f)
    fingerprint_dict_len = len(fingerprint_dict)
    print(f"Fingerprint dictionary length: {fingerprint_dict_len}")
    model = FusionCPI(fingerprint_dict_len, test_params)
    model.to(device)
    # 打印模型架构
    # print(model)
    prediction_list = []
    for i in range(num_batches):
        batch_indices = list(range(i * batch_size, min((i + 1) * batch_size, len(compounds))))

        # 切 batch
        batch_uniprotid     = [uniprotid_list[j] for j in batch_indices]
        batch_ID           = [ID_list[j] for j in batch_indices]
        batch_label        = [label_list[j] for j in batch_indices] 
        batch_compounds     = [compounds[j] for j in batch_indices]
        batch_adjacencies   = [adjacencies[j] for j in batch_indices]

        # 处理 protein embeddings
        prot_embed, prot_mask = process_protein_embeddings(batch_uniprotid, uniprotid_prot_embed_dict, device)

        # 构造 data_list
        data_list = [
            np.array(batch_compounds, dtype=object),
            np.array(batch_adjacencies, dtype=object),
        ]
        atoms_pad, atoms_mask, adjacencies_pad = batch2tensor_test(data_list, device)

        batch_data = (atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask)

        print(f"\n=== Running batch {i+1}/{num_batches} (size={len(batch_indices)}) ===")
        print(f"Batch UniProt IDs: {batch_uniprotid}")
        print(f"Batch IDs: {batch_ID}")
        print(f"Batch Labels: {batch_label}")
        prediction = test_model(model, test_params.best_model, batch_data, device)
        prediction_list.extend(prediction)    
    # 测试集参数计算
    # 转移到 CPU 并转换为 numpy 数组
    label_list = np.array(label_list, dtype=float)
    prediction_list = np.array([pred.cpu().numpy() for pred in prediction_list], dtype=float).squeeze()
    prediction_metrics = regression_scores_full(label_list, prediction_list)
    for metric, value in prediction_metrics.items():
        print(f"{metric}: {value:.4f}")

# 用于整个测试集的评估，Test_model更适合用于少量数据的测试
'''
python -m test.Test_model > casf_all_285_test.txt 2>&1
'''