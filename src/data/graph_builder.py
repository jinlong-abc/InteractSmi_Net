import numpy as np
from rdkit import Chem
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class MolecularGraphBuilder:
    def __init__(self, load_dicts: Dict[str, str] = None):
        if load_dicts:
            self.atom_dict = load_dicts['atom_dict']
            self.bond_dict = load_dicts['bond_dict']
            self.fingerprint_dict = load_dicts['fingerprint_dict']
            self.edge_dict = load_dicts['edge_dict']
        else:
            self.atom_dict = defaultdict(lambda: len(self.atom_dict))
            self.bond_dict = defaultdict(lambda: len(self.bond_dict))
            self.fingerprint_dict = defaultdict(lambda: len(self.fingerprint_dict))
            self.edge_dict = defaultdict(lambda: len(self.edge_dict))
        
    @staticmethod
    def load_dicts(prefix: str = '') -> Dict[str, Any]:
        import pickle
        with open(f'{prefix}/atom_dict.pkl', 'rb') as f:
            atom_dict = pickle.load(f)
        with open(f'{prefix}/bond_dict.pkl', 'rb') as f:
            bond_dict = pickle.load(f)
        with open(f'{prefix}/fingerprint_dict.pkl', 'rb') as f:
            fingerprint_dict = pickle.load(f)

        edge_dict = {}
        return {
            'atom_dict': atom_dict,
            'bond_dict': bond_dict,
            'fingerprint_dict': fingerprint_dict,
            'edge_dict': edge_dict
        }

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
    
    def atom_features(self, atoms: np.ndarray, i_jbond_dict: Dict, radius: int = 2) -> np.ndarray:
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict
            
            for _ in range(radius):
                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints
                
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
            atom_features = self.atom_features(atoms, i_jbond_dict, radius)

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
    
    def get_statistics(self) -> Dict[str, int]:

        return {
            'atom_types': len(self.atom_dict),
            'bond_types': len(self.bond_dict),
            'fingerprint_types': len(self.fingerprint_dict),
            'edge_types': len(self.edge_dict)
        }

