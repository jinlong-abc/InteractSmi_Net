import numpy as np
from rdkit import Chem
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pickle

class MolecularGraphBuilder_TEST:
    def __init__(self, fingerprint_dict_path, atom_dict_path, bond_dict_path):
        with open(bond_dict_path, 'rb') as f1:
            self.bond_dict = pickle.load(f1)
        with open(atom_dict_path, 'rb') as f2:
            self.atom_dict = pickle.load(f2)
        with open(fingerprint_dict_path, 'rb') as f3:
            self.fingerprint_dict = pickle.load(f3)
        self.fingerprint_dict_len = len(self.fingerprint_dict)

    def create_atoms(self, mol: Chem.Mol) -> np.ndarray:
        atoms = [a.GetSymbol() for a in mol.GetAtoms()] 
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')  
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)
    
    def create_ijbonddict(self, mol: Chem.Mol) -> Dict[int, List[Tuple[int, int]]]:
        i_jbond_dict = defaultdict(lambda: [])
        
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        
        atoms_set = set(range(mol.GetNumAtoms()))
        isolate_atoms = atoms_set - set(i_jbond_dict.keys())
        bond = self.bond_dict['nan']  
        for a in isolate_atoms:
            i_jbond_dict[a].append((a, bond))
            
        return i_jbond_dict

    def atom_features(self, atoms: np.ndarray, i_jbond_dict: Dict) -> np.ndarray:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        
        fingerprints = []
        for i, j_edge in i_jedge_dict.items():
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            fingerprint = (nodes[i], tuple(sorted(neighbors)))
            fingerprints.append(self.fingerprint_dict[fingerprint])
        
        return np.array(fingerprints)
    
    def create_adjacency(self, mol: Chem.Mol) -> np.ndarray:
        adjacency = Chem.GetAdjacencyMatrix(mol)
        adjacency = np.array(adjacency)
        adjacency += np.eye(adjacency.shape[0], dtype=int)
        return adjacency
    
    def mol_to_graph(self, smiles: str, radius: int = 2) -> Dict[str, np.ndarray]:
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                raise ValueError(f"无法解析SMILES: {smiles}")
            atoms = self.create_atoms(mol)
            i_jbond_dict = self.create_ijbonddict(mol)
            atom_features = self.atom_features(atoms, i_jbond_dict)
            adjacency = self.create_adjacency(mol)
            
            return {
                'atoms': atom_features,
                'adjacency': adjacency
            }
            
        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            return None
    
    def batch_mol_to_graph(self, smiles_list: List[str], radius: int = 2) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        compounds = []
        adjacencies = []
        
        for i, smiles in enumerate(smiles_list):
            graph_data = self.mol_to_graph(smiles, radius)
            if graph_data is not None:
                compounds.append(graph_data['atoms'])
                adjacencies.append(graph_data['adjacency'])
            else:
                print(f"跳过无效的SMILES: {smiles}")
                
        return compounds, adjacencies
    