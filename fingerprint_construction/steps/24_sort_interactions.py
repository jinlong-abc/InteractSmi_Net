"""Step 24: sort the relative IntSeq by residue position.

Re-orders the ``{aa  occurrence  interaction}`` tokens of ``IntSeq_2`` by their
numeric occurrence, producing the final interaction-fusion sequence table.

Output: ``bindingnet_pdbbind_final.csv``.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from tqdm import tqdm

_TAG = re.compile(r"\{([^}]+)\}")


def sort_intseq(text):
    if pd.isna(text):
        return ""
    processed = []
    for entry in _TAG.findall(str(text)):
        parts = entry.split()
        if len(parts) != 3:
            continue
        try:
            pos = int(parts[1])
        except ValueError:
            continue
        processed.append((pos, f"{{{parts[0]} {parts[1]} {parts[2]}}}"))
    processed.sort(key=lambda x: x[0])
    return " ".join(item[1] for item in processed)


def main():
    df = pd.read_csv(config.VDW_FILTERED)
    df["IntSeq_2"] = [sort_intseq(v) for v in tqdm(df["IntSeq_2"], desc="Sorting IntSeq_2")]

    columns = ["ID", "UniProt", "Assay Result", "Normal_SMILES", "Mapped_SMILES",
               "SMILES", "Fusion_SMILES", "Fusion_SMILES_2", "IntSeq_1",
               "IntSeq_2", "Sequence"]
    df = df[columns]
    df.to_csv(config.FINAL_OUTPUT, index=False)
    print(f"Wrote final table ({len(df)} rows) to {config.FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
