"""Step 10: derive AtomIdx-marked SMILES from the prepared ligand PDBs.

For each ligand, reads the PDB and writes both the plain canonical SMILES and
an atom-mapped SMILES whose atom-map numbers equal the RDKit atom indices.

Output: ``pdbbind_mapped_smiles.csv`` with columns
``PDBID, Normal_SMILES, Mapped_SMILES``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from rdkit import Chem


def main():
    rows = []
    errors = []

    for pdbid in sorted(os.listdir(config.COMPLEX_DIR)):
        pdb_file = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_ligand.pdb")
        if not os.path.isfile(pdb_file):
            continue

        mol = Chem.MolFromPDBFile(pdb_file)
        if mol is None:
            errors.append(pdb_file)
            continue

        normal_smiles = Chem.MolToSmiles(mol)

        mol_with_map = Chem.Mol(mol)
        for atom in mol_with_map.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        mapped_smiles = Chem.MolToSmiles(mol_with_map)

        rows.append((pdbid, normal_smiles, mapped_smiles))

    with open(config.MAPPED_SMILES, "w") as f:
        f.write("PDBID,Normal_SMILES,Mapped_SMILES\n")
        for pdbid, normal, mapped in rows:
            f.write(f"{pdbid},{normal},{mapped}\n")

    print(f"Wrote {len(rows)} rows; {len(errors)} unreadable ligands.")


if __name__ == "__main__":
    main()
