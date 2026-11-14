"""Step 16: merge the PDBBind IntSeq table with optional BindingNet tables.

Concatenates ``final_pdbbind_intseqs.csv`` (step 15) with the optional
BindingNet high/low/moderate files configured in ``config``. All inputs share
the same 7 columns (``ID, UniProt, Assay Result, Normal_SMILES, Mapped_SMILES,
IntSeq_1, Sequence``). BindingNet files are skipped when not set.

Output: ``bindingnet_pdbbind_intseqs_all.csv``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd


def main():
    frames = [pd.read_csv(config.FINAL_INTSEQ)]

    for path, tag in (
        (config.BINDINGNET_HIGH, "high"),
        (config.BINDINGNET_LOW, "low"),
        (config.BINDINGNET_MODERATE, "moderate"),
    ):
        if path and os.path.exists(path):
            frames.append(pd.read_csv(path))
            print(f"  merged BindingNet {tag}: {path}")
        elif path:
            print(f"  [skip] BindingNet {tag} not found: {path}")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(config.MERGED_INTSEQ, index=False)
    print(f"Merged {len(merged)} rows total.")


if __name__ == "__main__":
    main()
