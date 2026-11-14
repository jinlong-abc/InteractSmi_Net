"""Step 18: drop overly large ligands (long peptides in particular).

Removes ligands whose molecular weight exceeds the threshold, with the aim of
filtering out long peptide ligands.

Output: ``bindingnet_pdbbind_intseqs_all_filtered_mw.csv``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

MW_THRESHOLD = 1000
PEPTIDE_MW_THRESHOLD = 1000


def is_long_peptide(mol, min_amides=4, min_length=8):
    if not mol:
        return False
    amide_count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if (a1.GetSymbol() == "C" and a2.GetSymbol() == "N"
                    and any(n.GetSymbol() == "O" for n in a1.GetNeighbors())):
                amide_count += 1
    residue_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)N")))
    return amide_count >= min_amides or residue_count >= min_length


def main():
    df = pd.read_csv(config.SIM_FILTERED)
    keep, skipped = [], 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering by MW"):
        try:
            mol = Chem.MolFromSmiles(row["Normal_SMILES"])
            if mol is None:
                keep.append(row)
                continue
            mw = Descriptors.MolWt(mol)
            limit = PEPTIDE_MW_THRESHOLD if is_long_peptide(mol) else MW_THRESHOLD
            if mw <= limit:
                keep.append(row)
            else:
                skipped += 1
        except Exception:
            keep.append(row)

    out = pd.DataFrame(keep)
    out.to_csv(config.MW_FILTERED, index=False)
    print(f"Kept {len(out)} rows; skipped {skipped} large ligands.")


if __name__ == "__main__":
    main()
