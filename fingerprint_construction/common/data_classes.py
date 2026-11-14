"""Shared data classes used across pipeline steps.

These classes are pickled by step 05 and re-loaded by steps 06, 08 and 09.
Keeping them in a single importable module (rather than re-defining them in
each script) guarantees that ``pickle`` round-trips cleanly.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


class Residue:
    """A protein amino-acid residue (no atom information kept)."""

    def __init__(self, res_id: int, res_name: str):
        self.res_id = res_id          # residue number in the PDB file
        self.res_name = res_name      # three-letter name, e.g. "MET"

    def __repr__(self):
        return f"Residue({self.res_name} {self.res_id})"


class Chain:
    """A protein chain, holding its residues in order."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.residues: List[Residue] = []

    def add_residue(self, residue: Residue):
        self.residues.append(residue)

    def get_sequence(self) -> str:
        """Three-letter sequence joined by ``-``, e.g. ``MET-LYS-ALA``."""
        return "-".join(res.res_name for res in self.residues)

    def __repr__(self):
        return f"Chain({self.chain_id}, Residues: {len(self.residues)})"


class Atom:
    """A ligand atom: name + Cartesian coordinates."""

    def __init__(self, name: str, coords: Tuple[float, float, float]):
        self.name = name
        self.coords = coords

    def __repr__(self):
        return f"Atom(name={self.name}, coords={self.coords})"


class Ligand:
    """A ligand (HETATM group) with its atoms."""

    def __init__(self, name: str, chain: str):
        self.name = name
        self.chain = chain
        self.atoms: List[Atom] = []

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)

    def __repr__(self):
        return f"Ligand({self.name}, Atoms: {len(self.atoms)})"


class ProteinComplex:
    """A protein-ligand complex: protein chains + ligand(s)."""

    def __init__(self, data_id: str):
        self.data_id = data_id
        self.chains: Dict[str, Chain] = {}
        self.ligands: List[Ligand] = []

    def add_chain(self, chain: Chain):
        self.chains[chain.chain_id] = chain

    def add_ligand(self, ligand: Ligand):
        self.ligands.append(ligand)

    def get_protein_sequences(self) -> Dict[str, str]:
        return {cid: chain.get_sequence() for cid, chain in self.chains.items()}

    def __repr__(self):
        return f"ProteinComplex({self.data_id}, Chains: {len(self.chains)}, Ligands: {len(self.ligands)})"


@dataclass
class ProteinLigandInteractionClosest:
    """A closest-atoms interaction record (step 08)."""

    chemblid_all: str
    interact_ligand_atom_serial_numbers: List[int]
    interact_residue_pdb_ids: List[int]
    distances: List[float]
    chain_ids: List[str]
    uniprot_idx_list: List[int]


@dataclass
class ProteinLigandInteractionProlif:
    """A proLIF interaction record (steps 07 / 09)."""

    chemblid_all: str
    ligand_indices: List[int]
    protein_residues: List[int]
    interaction_types: List[str]
    residues_chain: List[str]
    uniprot_idx_list: List[int]
