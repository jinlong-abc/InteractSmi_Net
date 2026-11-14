"""Step 20: clean the IntSeq (drop the ligand atom index, sort by residue).

Rewrites each ``{atom  aa  position  interaction}`` token to
``{aa  position  interaction}`` (dropping the ligand atom number), removes
tokens whose ligand atom index falls outside the SMILES, and re-orders tokens
by residue position.

Output: ``bindingnet_pdbbind_intseqs_atomidx_filtered.csv``.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from tqdm import tqdm


def get_max_atom_idx(mapped_smiles):
    if pd.isna(mapped_smiles):
        return 0
    matches = re.findall(r":(\d+)\]", str(mapped_smiles))
    return max(int(i) for i in matches) if matches else 0


def clean_intseq(intseq_str, max_atom_idx):
    if pd.isna(intseq_str):
        return ""
    processed = []
    for entry in re.findall(r"\{(.*?)\}", str(intseq_str)):
        parts = entry.strip().split()
        if len(parts) != 4:
            continue
        try:
            atom_idx, aa_index = int(parts[0]), int(parts[2])
        except ValueError:
            continue
        if atom_idx > max_atom_idx + 1:
            continue
        processed.append((aa_index, f"{{{parts[1]} {aa_index} {parts[3]}}}"))

    processed.sort(key=lambda x: x[0])
    return " ".join(item[1] for item in processed)


def main():
    df = pd.read_csv(config.FUSION_SMILES_CSV)
    new_intseq = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning IntSeq"):
        max_idx = get_max_atom_idx(row["Mapped_SMILES"])
        new_intseq.append(clean_intseq(row["IntSeq_1"], max_idx))

    df["IntSeq_1"] = new_intseq
    df.to_csv(config.ATOM_IDX_FILTERED, index=False)
    print(f"Cleaned IntSeq for {len(df)} rows.")


if __name__ == "__main__":
    main()
