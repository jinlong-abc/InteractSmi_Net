import numpy as np
from rdkit import Chem
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pickle

class MolecularGraphBuilder_TEST:
    def __init__(self, fingerprint_dict_path, atom_dict_path, bond_dict_path):
        """初始化字典"""
        with open(bond_dict_path, 'rb') as f1:
            self.bond_dict = pickle.load(f1)
        with open(atom_dict_path, 'rb') as f2:
            self.atom_dict = pickle.load(f2)
        with open(fingerprint_dict_path, 'rb') as f3:
            self.fingerprint_dict = pickle.load(f3)
        self.fingerprint_dict_len = len(self.fingerprint_dict)

    def create_atoms(self, mol: Chem.Mol) -> np.ndarray:
        """
        mol : Chem.Mol
            RDKit分子对象
        ================
        np.ndarray
            原子特征数组
        """
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]  # 获取元素符号列表
        # 标记芳香原子
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')  # 将芳香原子标记为元组
        # 将原子映射到字典索引
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)
    
    def create_ijbonddict(self, mol: Chem.Mol) -> Dict[int, List[Tuple[int, int]]]:
        """
        mol : Chem.Mol
            RDKit分子对象
        ==============================================================
        Dict[int, List[Tuple[int, int]]]
            键连接字典，key是原子索引，value是(邻居原子索引, 键类型)的列表
        """
        i_jbond_dict = defaultdict(lambda: [])
        
        # 遍历所有化学键
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        
        # 处理孤立原子
        atoms_set = set(range(mol.GetNumAtoms()))
        isolate_atoms = atoms_set - set(i_jbond_dict.keys())
        bond = self.bond_dict['nan']  # 孤立原子的虚拟键
        for a in isolate_atoms:
            i_jbond_dict[a].append((a, bond))
            
        return i_jbond_dict

    def atom_features(self, atoms: np.ndarray, i_jbond_dict: Dict) -> np.ndarray:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        
        fingerprints = []
        for i, j_edge in i_jedge_dict.items():
            # 收集邻居信息
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            # 生成当前节点的指纹：(中心节点, 排序后的邻居信息)
            fingerprint = (nodes[i], tuple(sorted(neighbors)))
            fingerprints.append(self.fingerprint_dict[fingerprint])
        
        return np.array(fingerprints)
    
    def create_adjacency(self, mol: Chem.Mol) -> np.ndarray:
        """
        mol : Chem.Mol
            RDKit分子对象
        ======================
        np.ndarray
            邻接矩阵(包含自连接)
        """
        adjacency = Chem.GetAdjacencyMatrix(mol)
        adjacency = np.array(adjacency)
        # 添加自连接对角线为1
        adjacency += np.eye(adjacency.shape[0], dtype=int)
        return adjacency
    
    def mol_to_graph(self, smiles: str, radius: int = 2) -> Dict[str, np.ndarray]:
        """
        smiles : str
            SMILES字符串
        radius : int, default=2
            原子特征聚合半径
        =================================
        Dict[str, np.ndarray]
            包含'atoms', 'adjacency'的字典
        """
        try:
            # 从SMILES创建分子并加氢
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")
                
            # 生成原子特征
            atoms = self.create_atoms(mol)
            i_jbond_dict = self.create_ijbonddict(mol)
            atom_features = self.atom_features(atoms, i_jbond_dict)
            
            # 生成邻接矩阵
            adjacency = self.create_adjacency(mol)
            
            return {
                'atoms': atom_features,
                'adjacency': adjacency
            }
            
        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            return None
    
    def batch_mol_to_graph(self, smiles_list: List[str], radius: int = 2) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        smiles_list : List[str]
            SMILES字符串列表
        radius : int, default=2
            原子特征聚合半径
        =========================================
        Tuple[List[np.ndarray], List[np.ndarray]]
            (原子特征列表, 邻接矩阵列表)
        """
        compounds = []
        adjacencies = []
        
        for i, smiles in enumerate(smiles_list):
            # if i % 1000 == 0:
            # print(f"处理进度: {i}/{len(smiles_list)}")
                
            graph_data = self.mol_to_graph(smiles, radius)
            if graph_data is not None:
                compounds.append(graph_data['atoms'])
                adjacencies.append(graph_data['adjacency'])
            else:
                # 处理失败的情况，可以选择跳过或使用默认值
                print(f"跳过无效的SMILES: {smiles}")
                
        return compounds, adjacencies
    


# 使用示例
if __name__ == "__main__":
    import pickle
    builder = MolecularGraphBuilder_TEST()
    smiles_list = [
        "CCO",  
        "C1=CC=CC=C1",  
    ]
    compounds, adjacencies = builder.batch_mol_to_graph(smiles_list, radius=1)
    with open('/home/wjl/data/DTI_prj/Arch_Lab/DecoderOptionalArch/datasets/Pdbbind_512/atom_dict.pkl', 'rb') as f:
        fingerprint_dict = pickle.load(f)
        fingerprint_dict_len = len(fingerprint_dict)
        print(fingerprint_dict_len)
    # print(f"成功处理 {len(compounds)} 个分子")
    # print("字典统计:", builder.get_statistics())
    # builder.save_dictionaries("./dictionaries")