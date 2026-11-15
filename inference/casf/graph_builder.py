#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is adapted from:
Zhangli Lu & Luke Berg. (2023). 
BACPI: A bi-directional attention neural network for compound–protein interaction and binding affinity prediction [Computer software]. 
GitHub. https://github.com/CSUBioGroup/BACPI
"""
import numpy as np
from rdkit import Chem
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class MolecularGraphBuilder:
    def __init__(self, load_dicts: Dict[str, str] = None):
        """初始化字典"""
        if load_dicts:
            self.atom_dict = load_dicts['atom_dict']
            self.bond_dict = load_dicts['bond_dict']
            self.fingerprint_dict = load_dicts['fingerprint_dict']
            self.edge_dict = load_dicts['edge_dict']
        else:
            self.atom_dict = defaultdict(lambda: len(self.atom_dict))
            self.bond_dict = defaultdict(lambda: len(self.bond_dict))
            self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
            self.edge_dict = defaultdict(lambda: len(self.edge_dict)) # radius>1时用于更新边
        
    @staticmethod
    def load_dicts(prefix: str = '') -> Dict[str, Any]:
        import pickle
        with open(f'{prefix}/atom_dict.pkl', 'rb') as f:
            atom_dict = pickle.load(f)
        with open(f'{prefix}/bond_dict.pkl', 'rb') as f:
            bond_dict = pickle.load(f)
        with open(f'{prefix}/fingerprint_dict.pkl', 'rb') as f:
            fingerprint_dict = pickle.load(f)
        # with open(f'{prefix}/edge_dict.pkl', 'rb') as f:
        #     edge_dict = pickle.load(f)
        edge_dict = {}
        return {
            'atom_dict': atom_dict,
            'bond_dict': bond_dict,
            'fingerprint_dict': fingerprint_dict,
            'edge_dict': edge_dict
        }
        # self.atom_dict = defaultdict(lambda: len(self.atom_dict))
        # self.bond_dict = defaultdict(lambda: len(self.bond_dict))
        # self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
        # self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        
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
    
    def atom_features(self, atoms: np.ndarray, i_jbond_dict: Dict, radius: int = 1) -> np.ndarray:
        """
        基于邻域信息生成原子特征指纹
        atoms : np.ndarray
            原子数组
        i_jbond_dict : Dict
            键连接字典
        radius : int, default=2
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
                # _i_jedge_dict = defaultdict(lambda: [])
                # for i, j_edge in i_jedge_dict.items():
                #     for j, edge in j_edge:
                #         both_side = tuple(sorted((nodes[i], nodes[j])))
                #         edge = self.edge_dict[(both_side, edge)]
                #         _i_jedge_dict[i].append((j, edge))
                # i_jedge_dict = _i_jedge_dict
        
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

# 使用示例
if __name__ == "__main__":
    builder = MolecularGraphBuilder()
    smiles_list = [
        "CCO",  
        "C1=CC=CC=C1",  
    ]
    compounds, adjacencies = builder.batch_mol_to_graph(smiles_list)
    print(f"成功处理 {len(compounds)} 个分子")
    print("字典统计:", builder.get_statistics())
    builder.save_dictionaries("./dictionaries")