"""Step 08: find closest-atoms interactions between protein and ligand.

For each residue, finds the closest ligand atom within 5.0 A (residue center of
mass vs. ligand atom coordinates), then maps the residue to its UniProt index
via the SSW alignment. Output pickle: ``closest_interactions.pkl``.
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import numpy as np
from Bio.PDB import PDBParser

from common.data_classes import ProteinLigandInteractionClosest
from common.ssw import get_result_dict, get_pdb_to_uniprot_map

_EXCLUDED_RESIDUES = {"HOH", "WAT", "NA", "CL", "MG", "CA", "K", "ZN"}


def calculate_center_of_residue(residue):
    return np.mean([atom.coord for atom in residue.get_atoms()], axis=0)


def find_closest_atoms_to_ligand(protein_pdb_file, ligand_pdb_file, distance_cutoff=5.0):
    parser = PDBParser(QUIET=True)
    protein_model = parser.get_structure("protein", protein_pdb_file)[0]
    ligand_model = parser.get_structure("ligand", ligand_pdb_file)[0]

    ligand_atoms = []
    for chain in ligand_model:
        for residue in chain:
            for atom in residue:
                if atom.element != "H":
                    ligand_atoms.append(atom)

    serial_numbers, residue_ids, distances, chain_ids = [], [], [], []
    for chain in protein_model:
        for residue in chain:
            res_name = residue.get_resname().strip()
            if residue.id[0] == " " and res_name not in _EXCLUDED_RESIDUES:
                center = calculate_center_of_residue(residue)
                min_dist, closest = float("inf"), None
                for ligand_atom in ligand_atoms:
                    d = float(np.linalg.norm(center - ligand_atom.coord))
                    if d < min_dist and d <= distance_cutoff:
                        min_dist, closest = d, ligand_atom
                if closest is not None:
                    serial_numbers.append(closest.serial_number)
                    residue_ids.append(residue.id[1])
                    distances.append(min_dist)
                    chain_ids.append(chain.id)

    return serial_numbers, residue_ids, distances, chain_ids


def main():
    with open(config.ALL_PDBS_PKL, "rb") as f:
        pdb_data = pickle.load(f)

    result_dict = get_result_dict(config.SEQ_ALIGN)
    pdb_to_uniprot_map = get_pdb_to_uniprot_map(result_dict)

    interactions = []
    for pdbid in os.listdir(config.COMPLEX_DIR):
        ligand_pdb = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_ligand.pdb")
        protein_pdb = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_protein.pdb")
        if not (os.path.isfile(ligand_pdb) and os.path.isfile(protein_pdb)):
            continue

        serial_numbers, residue_ids, distances, chain_ids = find_closest_atoms_to_ligand(
            protein_pdb, ligand_pdb
        )
        if not chain_ids or pdbid not in pdb_data or pdbid not in pdb_to_uniprot_map:
            continue

        # main chain = chain with the most interacting residues
        main_chain = max(set(chain_ids), key=chain_ids.count)

        res_id_list, res_name_list = [], []
        for chain in pdb_data[pdbid].chains.values():
            if chain.chain_id == main_chain:
                res_id_list = [r.res_id for r in chain.residues]
                res_name_list = [r.res_name for r in chain.residues]

        if main_chain not in pdb_to_uniprot_map[pdbid]:
            continue
        chain_map = pdb_to_uniprot_map[pdbid][main_chain]

        uniprot_idx_list = []
        for rid in residue_ids:
            if rid in res_id_list:
                idx = res_id_list.index(rid)
                if idx < len(chain_map):
                    uniprot_idx_list.append(chain_map[idx])

        interactions.append(
            ProteinLigandInteractionClosest(
                chemblid_all=pdbid,
                interact_ligand_atom_serial_numbers=serial_numbers,
                interact_residue_pdb_ids=residue_ids,
                distances=distances,
                chain_ids=chain_ids,
                uniprot_idx_list=uniprot_idx_list,
            )
        )

    with open(config.CLOSEST_PKL, "wb") as f:
        pickle.dump(interactions, f)
    print(f"Closest-atoms complete: {len(interactions)} complexes.")


if __name__ == "__main__":
    main()
