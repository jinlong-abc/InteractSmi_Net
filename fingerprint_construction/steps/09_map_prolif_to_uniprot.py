"""Step 09: map proLIF interactions onto the UniProt sequence.

Takes the proLIF interactions from step 07 and translates each PDB residue
number to its UniProt index using the SSW alignment. Output pickle:
``prolif_interactions_mapped.pkl`` (ready for the prolif_closest merge).
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from common.data_classes import ProteinLigandInteractionProlif
from common.ssw import get_result_dict, get_pdb_to_uniprot_map


def main():
    with open(config.ALL_PDBS_PKL, "rb") as f:
        pdb_data = pickle.load(f)

    result_dict = get_result_dict(config.SEQ_ALIGN)
    pdb_to_uniprot_map = get_pdb_to_uniprot_map(result_dict)

    with open(config.PROLIF_PKL, "rb") as f:
        prolif_interactions = pickle.load(f)

    interactions = []
    for inter in prolif_interactions:
        pdbid = inter.chemblid_all
        if not inter.residues_chain or pdbid not in pdb_data or pdbid not in pdb_to_uniprot_map:
            continue

        main_chain = max(set(inter.residues_chain), key=inter.residues_chain.count)

        res_id_list = []
        for chain in pdb_data[pdbid].chains.values():
            if chain.chain_id == main_chain:
                res_id_list = [r.res_id for r in chain.residues]

        if main_chain not in pdb_to_uniprot_map[pdbid]:
            continue
        chain_map = pdb_to_uniprot_map[pdbid][main_chain]

        uniprot_idx_list = []
        for rid in inter.protein_residues:
            if rid in res_id_list:
                idx = res_id_list.index(rid)
                if idx < len(chain_map):
                    uniprot_idx_list.append(chain_map[idx])

        interactions.append(
            ProteinLigandInteractionProlif(
                chemblid_all=pdbid,
                ligand_indices=inter.ligand_indices,
                protein_residues=inter.protein_residues,
                interaction_types=inter.interaction_types,
                residues_chain=inter.residues_chain,
                uniprot_idx_list=uniprot_idx_list,
            )
        )

    with open(config.PROLIF_MAPPED_PKL, "wb") as f:
        pickle.dump(interactions, f)
    print(f"Mapped {len(interactions)} proLIF interactions to UniProt.")


if __name__ == "__main__":
    main()
