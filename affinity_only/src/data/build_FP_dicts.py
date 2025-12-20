import numpy as np
from rdkit import Chem
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class MolecularGraphBuilder:
    def __init__(self):
        """初始化字典"""
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        
    def create_atoms(self, mol: Chem.Mol) -> np.ndarray:
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
    
    def atom_features(self, atoms: np.ndarray, i_jbond_dict: Dict, radius: int = 1) -> np.ndarray:
        """
        基于邻域信息生成原子特征指纹
        atoms : np.ndarray
            原子数组
        i_jbond_dict : Dict
            键连接字典
        radius : int, default=1
            邻域半径
        ========================
        np.ndarray
            原子特征指纹数组
        """
        if (len(atoms) == 1) or (radius == 0):
            # 单原子分子或半径为0时，直接使用原子特征
            fingerprints = [self.fingerprint_dict[a] for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict
            
            # 迭代聚合邻域信息
            for _ in range(radius):
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    # 收集邻居信息
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    # 生成当前节点的指纹：(中心节点, 排序后的邻居信息)
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                
                # 更新节点特征
                nodes = fingerprints
                
                # 更新边信息
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict
        
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
        # 添加自连接
        adjacency += np.eye(adjacency.shape[0], dtype=int)
        return adjacency
    
    def mol_to_graph(self, smiles: str, radius: int = 1) -> Dict[str, np.ndarray]:
        """
        smiles : str
            SMILES字符串
        radius : int, default=1
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
            atom_features = self.atom_features(atoms, i_jbond_dict, radius)
            
            # 生成邻接矩阵
            adjacency = self.create_adjacency(mol)
            
            return {
                'atoms': atom_features,
                'adjacency': adjacency
            }
            
        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            return None
    
    def batch_mol_to_graph(self, smiles_list: List[str], radius: int = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        smiles_list : List[str]
            SMILES字符串列表
        radius : int, default=1
            原子特征聚合半径
        =========================================
        Tuple[List[np.ndarray], List[np.ndarray]]
            (原子特征列表, 邻接矩阵列表)
        """
        compounds = []
        adjacencies = []
        
        for i, smiles in enumerate(smiles_list):
            # if i % 1000 == 0:
            #     print(f"处理进度: {i}/{len(smiles_list)}")
                
            graph_data = self.mol_to_graph(smiles, radius)
            if graph_data is not None:
                compounds.append(graph_data['atoms'])
                adjacencies.append(graph_data['adjacency'])
            else:
                # 处理失败的情况，可以选择跳过或使用默认值
                print(f"跳过无效的SMILES: {smiles}")
                
        return compounds, adjacencies
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取字典统计信息
        --------
        Dict[str, int]
            各字典的大小
        """
        return {
            'atom_types': len(self.atom_dict),
            'bond_types': len(self.bond_dict),
            'fingerprint_types': len(self.fingerprint_dict),
            'edge_types': len(self.edge_dict)
        }
    
    def get_dictionaries(self) -> Dict[str, Dict[Any, int]]:
        """
        获取完整的字典映射
        -------------
        Dict[str, Dict[Any, int]]
            包含所有字典的字典
        """
        return {
            'atom_dict': dict(self.atom_dict),
            'bond_dict': dict(self.bond_dict),
            'fingerprint_dict': dict(self.fingerprint_dict),
            'edge_dict': dict(self.edge_dict)
        }


def build_dictionaries_from_smiles(smiles_list: List[str], radius: int = 1) -> Dict[str, Dict[Any, int]]:
    """
    直接从SMILES列表构建字典
    
    Parameters
    ----------
    smiles_list : List[str]
        SMILES字符串列表
    radius : int, default=1
        原子特征聚合半径
        
    Returns
    -------
    Dict[str, Dict[Any, int]]
        包含所有字典的字典
    """
    builder = MolecularGraphBuilder()
    
    # 处理所有SMILES来构建字典
    compounds, adjacencies = builder.batch_mol_to_graph(smiles_list, radius)
    
    # 返回构建好的字典
    return builder.get_dictionaries()


def build_dictionaries_and_graphs_from_smiles(smiles_list: List[str], radius: int = 1) -> Tuple[Dict[str, Dict[Any, int]], List[np.ndarray], List[np.ndarray]]:
    """
    从SMILES列表构建字典和图形数据
    
    Parameters
    ----------
    smiles_list : List[str]
        SMILES字符串列表
    radius : int, default=1
        原子特征聚合半径
        
    Returns
    -------
    Tuple[Dict[str, Dict[Any, int]], List[np.ndarray], List[np.ndarray]]
        (字典集合, 原子特征列表, 邻接矩阵列表)
    """
    builder = MolecularGraphBuilder()
    
    # 处理所有SMILES来构建字典和图形数据
    compounds, adjacencies = builder.batch_mol_to_graph(smiles_list, radius)
    
    # 返回构建好的字典和图形数据
    return builder.get_dictionaries(), compounds, adjacencies


# 使用示例
if __name__ == "__main__":
    # 示例SMILES列表
    # smiles_list = [
    #     "CCO",  # 乙醇
    #     "CC(=O)O",  # 乙酸
    #     "c1ccccc1",  # 苯
    #     "CNC",  # 二甲胺
    #     "CCOC(=O)C"  # 乙酸乙酯
    # ]
    import pandas as pd
    # df = pd.read_csv("/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/data/BindingDB_BindingNet_PDBbind_all/scripts/2_all_smiles.csv")
    # smiles_list = df['SMILES'].tolist()       
    # with open("/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/data/BindingDB_BindingNet_PDBbind_all/scripts/2_all_smiles.csv", "r") as f:
        # smiles_list = [line.strip().split(",")[0] for line in f.readlines()]
    # PDBbind
    df = pd.read_csv("/home/wjl/data/DTI_prj/Arch_Lab/without_decoder_apply/data/Pdbbind/1_Pdbbind_out.csv")
    smiles_list = df['Normal SMILES'].tolist()   

    # 方法1: 只获取字典
    print("=== 只获取字典 ===")
    dictionaries = build_dictionaries_from_smiles(smiles_list)
    import pickle
    atom_dict = dictionaries['atom_dict']
    bond_dict = dictionaries['bond_dict']
    fingerprint_dict = dictionaries['fingerprint_dict']
    edge_dict = dictionaries['edge_dict']
    with open("atom_dict.pkl", "wb") as f:
        pickle.dump(atom_dict, f)
    with open("bond_dict.pkl", "wb") as f:
        pickle.dump(bond_dict, f)
    with open("fingerprint_dict.pkl", "wb") as f:
        pickle.dump(fingerprint_dict, f)
    with open("edge_dict.pkl", "wb") as f:
        pickle.dump(edge_dict, f)

    print("原子字典:", dictionaries['atom_dict'])
    print("键字典:", dictionaries['bond_dict'])
    print("指纹字典大小:", len(dictionaries['fingerprint_dict']))
    print("边字典大小:", len(dictionaries['edge_dict']))
    
    print("\n=== 获取字典和图形数据 ===")
    # 方法2: 获取字典和图形数据
    # dictionaries, compounds, adjacencies = build_dictionaries_and_graphs_from_smiles(smiles_list)
    
    # print("处理了", len(compounds), "个分子")
    # print("原子特征形状示例:", compounds[0].shape)
    # print("邻接矩阵形状示例:", adjacencies[0].shape)
    
    # print("\n=== 字典统计 ===")
    # # 使用原始类的方法获取统计信息
    # builder = MolecularGraphBuilder()
    # builder.batch_mol_to_graph(smiles_list)
    # stats = builder.get_statistics()
    # print("字典统计:", stats)

# [11:00:32] SMILES Parse Error: Failed parsing SMILES 'smiles' for input: 'smiles'
# 处理SMILES smiles 时出错: Python argument types in
#     rdkit.Chem.rdmolops.AddHs(NoneType)
# did not match C++ signature:
#     AddHs(RDKit::ROMol mol, bool explicitOnly=False, bool addCoords=False, boost::python::api::object onlyOnAtoms=None, bool addResidueInfo=False)
# 跳过无效的SMILES: smiles
# 原子字典: {'C': 0, ('N', 'aromatic'): 1, ('C', 'aromatic'): 2, 'O': 3, 'N': 4, 'H': 5, 'Cl': 6, 'F': 7, ('O', 'aromatic'): 8, ('S', 'aromatic'): 9, 'Br': 10, 'S': 11, 'I': 12, 'P': 13, 'As': 14, 'Zn': 15, 'Na': 16, 'Si': 17, 'Se': 18, 'B': 19, 'Mn': 20, 'Sb': 21, 'Ru': 22, ('Se', 'aromatic'): 23, 'Hg': 24, 'Pt': 25, 'Fe': 26, 'Re': 27, 'Li': 28, 'V': 29, 'Au': 30, 'K': 31, 'W': 32, 'Pd': 33, 'Sn': 34, 'Al': 35, 'Cu': 36, 'Ag': 37, 'Nb': 38, ('Te', 'aromatic'): 39, 'Gd': 40, 'Te': 41, 'Ni': 42, 'Mo': 43, 'Co': 44, 'Ir': 45, 'Rh': 46, 'Mg': 47, 'Be': 48, 'Os': 49}
# 键字典: {'SINGLE': 0, 'AROMATIC': 1, 'DOUBLE': 2, 'nan': 3, 'TRIPLE': 4}
# 指纹字典大小: 1454
# 边字典大小: 10617