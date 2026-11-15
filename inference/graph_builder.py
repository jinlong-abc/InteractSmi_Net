"""Molecular-graph construction for the InteractSmi benchmark (radius = 1).

This reproduces the exact preprocessing used to build ``fingerprint_dict.pkl``
and the packed graph cache: for each atom the fingerprint key is

    (atom_symbol, tuple(sorted((neighbor_symbol, bond_type), key=repr)))

where aromatic atoms are marked ``(symbol, "aromatic")`` and bond types are the
RDKit names (``SINGLE``/``DOUBLE``/``AROMATIC``/...). ``fingerprint_dict.pkl``
maps these keys to the integer ids consumed by the model's ``Embedding`` layer.
"""

import numpy as np
from rdkit import Chem


def atom_symbol(atom):
    return (atom.GetSymbol(), "aromatic") if atom.GetIsAromatic() else atom.GetSymbol()


def build_graph(smiles, fingerprint_dict):
    """Return ``(fingerprint_ids, edges)`` for one SMILES string.

    ``fingerprint_ids`` are the per-atom integer ids (length = number of atoms,
    hydrogens included). ``edges`` is an ``(E, 2)`` int array of (src, dst)
    pairs including self-loops, matching the packed ``graph_nodes``/``graph_edges``
    format.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    atoms = [atom_symbol(a) for a in mol.GetAtoms()]
    neighbors = [[] for _ in atoms]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype = str(bond.GetBondType())
        neighbors[i].append((atoms[j], btype))
        neighbors[j].append((atoms[i], btype))

    fps = [(atoms[i], tuple(sorted(ns, key=repr))) for i, ns in enumerate(neighbors)]
    ids = np.asarray([fingerprint_dict[k] for k in fps], dtype=np.int64)

    adj = Chem.GetAdjacencyMatrix(mol).astype(np.int8)
    adj += np.eye(len(atoms), dtype=np.int8)
    edges = np.argwhere(adj).astype(np.int64)

    return ids, edges
