import numpy as np
from rdkit import Chem
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pickle

class MolecularGraphBuilder:
    """使用预训练字典构建分子图的类"""
    
    def __init__(self, dict_filepath: str = None):
        """
        初始化图构建器
        
        dict_filepath : str, optional
            预训练字典文件路径
        """
        self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        
        if dict_filepath:
            self.load_dictionaries(dict_filepath)
    
    def load_dictionaries(self, filepath: str) -> None:
        """加载预训练的字典"""
        with open(filepath, 'rb') as f:
            dict_data = pickle.load(f)
        
        # 更新字典（保持defaultdict的特性）
        self.atom_dict.update(dict_data['atom_dict'])
        self.bond_dict.update(dict_data['bond_dict'])
        self.fingerprint_dict.update(dict_data['fingerprint_dict'])
        self.edge_dict.update(dict_data['edge_dict'])
        
        print(f"字典加载完成，统计信息: {self.get_statistics()}")
    
    def create_atoms(self, mol: Chem.Mol) -> np.ndarray:
        """创建原子特征"""
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        
        # 使用预训练字典，未知原子返回0或特殊标记
        atoms = [self.atom_dict.get(a, 0) for a in atoms]
        return np.array(atoms)
    
    def create_ijbonddict(self, mol: Chem.Mol) -> Dict[int, List[Tuple[int, int]]]:
        """创建键连接字典"""
        i_jbond_dict = defaultdict(lambda: [])
        
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond_type = str(b.GetBondType())
            bond = self.bond_dict.get(bond_type, 0)  # 使用get避免KeyError
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        
        atoms_set = set(range(mol.GetNumAtoms()))
        isolate_atoms = atoms_set - set(i_jbond_dict.keys())
        bond = self.bond_dict.get('nan', 0)
        for a in isolate_atoms:
            i_jbond_dict[a].append((a, bond))
            
        return i_jbond_dict
    
    def atom_features(self, atoms: np.ndarray, i_jbond_dict: Dict, radius: int = 2) -> np.ndarray:
        """生成原子特征指纹"""
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict.get(a, 0) for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict
            
            for _ in range(radius):
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict.get(fingerprint, 0))
                
                nodes = fingerprints
                
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge_key = (both_side, edge)
                        edge_val = self.edge_dict.get(edge_key, 0)
                        _i_jedge_dict[i].append((j, edge_val))
                i_jedge_dict = _i_jedge_dict
        
        return np.array(fingerprints)
    
    def create_adjacency(self, mol: Chem.Mol) -> np.ndarray:
        """创建邻接矩阵"""
        adjacency = Chem.GetAdjacencyMatrix(mol)
        adjacency = np.array(adjacency)
        adjacency += np.eye(adjacency.shape[0], dtype=int)
        return adjacency
    
    def mol_to_graph(self, smiles: str, radius: int = 2) -> Dict[str, np.ndarray]:
        """将SMILES转换为图数据"""
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")
                
            atoms = self.create_atoms(mol)
            i_jbond_dict = self.create_ijbonddict(mol)
            atom_features = self.atom_features(atoms, i_jbond_dict, radius)
            adjacency = self.create_adjacency(mol)
            
            return {
                'atoms': atom_features,
                'adjacency': adjacency,
                'num_atoms': len(atom_features)
            }
            
        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            return None
    
    def batch_mol_to_graph(self, smiles_list: List[str], radius: int = 2) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """批量转换SMILES为图数据"""
        compounds = []
        adjacencies = []
        
        for smiles in smiles_list:
            graph_data = self.mol_to_graph(smiles, radius)
            if graph_data is not None:
                compounds.append(graph_data['atoms'])
                adjacencies.append(graph_data['adjacency'])
            else:
                print(f"跳过无效的SMILES: {smiles}")
                
        return compounds, adjacencies
    
    def get_statistics(self) -> Dict[str, int]:
        """获取字典统计信息"""
        return {
            'atom_types': len(self.atom_dict),
            'bond_types': len(self.bond_dict),
            'fingerprint_types': len(self.fingerprint_dict),
            'edge_types': len(self.edge_dict)
        }


# 使用示例
if __name__ == "__main__":
    # 初始化图构建器并加载预训练字典
    graph_builder = MolecularGraphBuilder("molecular_dictionaries.pkl")
    
    # 构建单个分子图
    smiles = "CCO"  # 乙醇
    graph_data = graph_builder.mol_to_graph(smiles)
    print(f"原子特征: {graph_data['atoms']}")
    print(f"邻接矩阵形状: {graph_data['adjacency'].shape}")
    
    # 批量构建
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]  # 乙醇、乙酸、苯
    compounds, adjacencies = graph_builder.batch_mol_to_graph(smiles_list)
    print(f"成功构建 {len(compounds)} 个分子图")