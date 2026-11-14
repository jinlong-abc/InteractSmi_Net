"""Step 17: remove near-duplicate ligands within each protein (Tanimoto).

Within each UniProt group, ligands whose Morgan fingerprints are more similar
than ``threshold`` are de-duplicated (the first occurrence is kept). This keeps
the dataset from being dominated by near-identical ligands of the same target.

Output: ``bindingnet_pdbbind_intseqs_all_filtered_sim.csv``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

THRESHOLD = 0.3


def tanimoto(smiles1, smiles2):
    try:
        m1 = Chem.MolFromSmiles(smiles1)
        m2 = Chem.MolFromSmiles(smiles2)
        if m1 is None or m2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def filter_similar(df, threshold=THRESHOLD):
    keep_rows = []
    for uniprot, group in tqdm(df.groupby("UniProt"), desc="Filtering similar ligands"):
        smiles = group["Normal_SMILES"].tolist()
        indices = list(group.index)
        keep = set(indices)

        for i in range(len(smiles)):
            if indices[i] not in keep:
                continue
            for j in range(i + 1, len(smiles)):
                if indices[j] not in keep:
                    continue
                if tanimoto(smiles[i], smiles[j]) > threshold:
                    keep.discard(indices[j])

        keep_rows.append(group.loc[sorted(keep)])

    return pd.concat(keep_rows)


def main():
    df = pd.read_csv(config.MERGED_INTSEQ)
    filtered = filter_similar(df, THRESHOLD)
    filtered.to_csv(config.SIM_FILTERED, index=False)
    print(f"Kept {len(filtered)} / {len(df)} rows after similarity filtering.")


if __name__ == "__main__":
    main()
