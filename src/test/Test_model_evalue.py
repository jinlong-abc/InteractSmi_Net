import torch
import torch
import os
import pickle
import math
from test.Test_GraphBuilder import MolecularGraphBuilder_TEST
from data.data_utils import *
from test.Test_Config import get_config
from model.model import FusionCPI
from torch.nn import functional as F

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

def load_protein_embeddings(params):
    """加载蛋白质编码"""
    print('Loading protein embeddings...')
    if not os.path.exists(params.prot_esmc_embed):
        raise FileNotFoundError(f"Protein embedding file not found: {params.prot_esmc_embed}")
    
    with open(params.prot_esmc_embed, "rb") as f:
        uniprotid_prot_embed_dict = pickle.load(f)
    
    print(f"Loaded embeddings for {len(uniprotid_prot_embed_dict)} proteins")
    return uniprotid_prot_embed_dict

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


def batch2tensor_test(batch_data, device, vocab=None, params=None):
    if params.multitask:
        atoms_pad, atoms_mask = batch_pad(batch_data[0])
        adjacencies_pad, _ = batch_pad(batch_data[1])
        Sequences_list = batch_data[2]
        Smiles_list = batch_data[3]

        padded_smiles, mask_smiles = tokenize_pad_fusionsmi(Smiles_list, vocab)
        padded_sequence, mask_sequence = tokenize_pad_prot_sequence(Sequences_list, vocab, \
                                                                    max_len=params.decoder_prot_max_len)
        atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
        atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
        adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
        Smiles_list = Variable(torch.LongTensor(padded_smiles)).to(device)
        Smiles_mask = Variable(torch.FloatTensor(mask_smiles)).to(device)
        Sequence_list = Variable(torch.LongTensor(padded_sequence)).to(device)
        Sequence_mask = Variable(torch.FloatTensor(mask_sequence)).to(device)

    else:
        atoms_pad, atoms_mask = batch_pad(batch_data[0])
        adjacencies_pad, _ = batch_pad(batch_data[1])

        atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
        atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
        adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
        # ====== 创建占位符 (dummy tensors)，保证返回值一致 ======
        Smiles_list   = torch.zeros((1, 1), dtype=torch.long, device=device)
        Smiles_mask   = torch.zeros((1, 1), dtype=torch.float, device=device)
        Sequence_list  = torch.zeros((1, 1), dtype=torch.long, device=device)
        Sequence_mask  = torch.zeros((1, 1), dtype=torch.float, device=device)

    return atoms_pad, atoms_mask, adjacencies_pad, Smiles_list, \
          Smiles_mask, Sequence_list, Sequence_mask

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

def test_model(model, checkpoint_path, batch_data, device, vocab, max_len=200):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint  
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        inputs = [x.to(device) if torch.is_tensor(x) else x for x in batch_data]
        if model.multitask:
            prediction, generated = model.inference(*inputs)
        else:
            prediction = model.inference(*inputs, max_len=max_len)
            generated = None
        print(f"Predicted affinity: {prediction.squeeze().cpu().numpy()}")

        # 可读形式
        if generated is not None:
            for i in range(generated.size(0)):
                tokens = []
                for idx in generated[i].cpu().tolist():
                    token = vocab['id2token'].get(idx, "<UNK>")
                    if token == "<EOS>":
                        break
                    if token not in ["<PAD>"]:
                        tokens.append(token)
                seq = " ".join(tokens)
                print(f"Generated InteractSmi {i+1}: {seq}")


if __name__ == '__main__':
    test_params = get_config()

    builder = MolecularGraphBuilder_TEST(
        test_params.fingerprint_dict_path, 
        test_params.atom_dict_path, 
        test_params.bond_dict_path
    )

    ID_list = []
    normal_smiles_list = []
    Smiles_list = []
    prot_sep_list = []
    uniprotid_list = []
    label_list = []
    with open(test_params.test_data, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split(',')
            ID_list.append(line[0])
            normal_smiles_list.append(line[4])
            Smiles_list.append(line[5])
            prot_sep_list.append(line[9])
            uniprotid_list.append(line[1])
            label_list.append(line[3])

    with open(test_params.vocab, "rb") as f:
        vocab = pickle.load(f)

    device = setup_environment(test_params)

    compounds, adjacencies = builder.batch_mol_to_graph(normal_smiles_list, radius=1)

    uniprotid_prot_embed_dict = load_protein_embeddings(test_params)

    batch_size = test_params.batch_size
    num_batches = math.ceil(len(compounds) / batch_size)
    print(f"Total samples: {len(compounds)}, Batch size: {batch_size}, Num batches: {num_batches}")

    with open(test_params.fingerprint_dict_path, 'rb') as f:
        fingerprint_dict = pickle.load(f)
    fingerprint_dict_len = len(fingerprint_dict)
    model = FusionCPI(fingerprint_dict_len, test_params, vocab)
    model.to(device)

    for i in range(num_batches):
        batch_indices = list(range(i * batch_size, min((i + 1) * batch_size, len(compounds))))

        batch_ID = [ID_list[j] for j in batch_indices]
        batch_normal_smiles = [normal_smiles_list[j] for j in batch_indices]
        batch_Smiles        = [Smiles_list[j] for j in batch_indices]
        batch_prot_sep      = [prot_sep_list[j] for j in batch_indices]
        batch_uniprotid     = [uniprotid_list[j] for j in batch_indices]

        batch_compounds     = [compounds[j] for j in batch_indices]
        batch_adjacencies   = [adjacencies[j] for j in batch_indices]

        prot_embed, prot_mask = process_protein_embeddings(batch_uniprotid, uniprotid_prot_embed_dict, device)

        data_list = [
            np.array(batch_compounds, dtype=object),
            np.array(batch_adjacencies, dtype=object),
            np.array(batch_prot_sep, dtype=object),
            np.array(batch_Smiles, dtype=object),
        ]
        atoms_pad, atoms_mask, adjacencies_pad, Smiles, Smiles_mask, Sequence, Sequence_mask = \
            batch2tensor_test(data_list, device, vocab=vocab, params=test_params)

        batch_data = (
            atoms_pad, atoms_mask, adjacencies_pad, prot_embed, prot_mask,
            Smiles, Smiles_mask, Sequence, Sequence_mask
        )

        test_model(model, test_params.best_model, batch_data, device, vocab)
