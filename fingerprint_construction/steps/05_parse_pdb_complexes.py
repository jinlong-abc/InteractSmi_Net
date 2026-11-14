"""Step 05: parse the prepared complex PDBs into structured objects.

Parses each ``*_complex.pdb`` produced by step 01 into a
``ProteinComplex`` (protein chains as residues, ligands as atoms) and pickles
the whole dataset to ``pdbbind_all_pdbs.pkl`` for the downstream steps.
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from Bio import PDB
from common.data_classes import ProteinComplex, Chain, Residue, Ligand, Atom


def parse_pdb_file(pdb_filename, data_name):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_filename)
    protein_complex = ProteinComplex(data_name)

    # Protein chains: standard amino-acid residues only.
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if any(res.get_id()[0] == " " for res in chain):
                protein_chain = Chain(chain_id)
                for residue in chain:
                    if residue.get_id()[0] == " ":
                        protein_chain.add_residue(
                            Residue(residue.get_id()[1], residue.get_resname())
                        )
                protein_complex.add_chain(protein_chain)

    # Ligands: HETATM groups, kept at atom level.
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != " ":
                    ligand = Ligand(residue.get_resname(), chain.get_id())
                    for atom in residue:
                        ligand.add_atom(Atom(atom.get_name(), tuple(atom.get_coord())))
                    protein_complex.add_ligand(ligand)

    return protein_complex


def main():
    pdb_data = {}
    for pdbid in os.listdir(config.COMPLEX_DIR):
        complex_pdb = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_complex.pdb")
        if os.path.isfile(complex_pdb):
            try:
                pdb_data[pdbid] = parse_pdb_file(complex_pdb, pdbid)
            except Exception as e:
                print(f"[Error] failed to parse {complex_pdb}: {e}")

    with open(config.ALL_PDBS_PKL, "wb") as f:
        pickle.dump(pdb_data, f)
    print(f"Pickled {len(pdb_data)} complexes to {config.ALL_PDBS_PKL}")


if __name__ == "__main__":
    main()
