"""Step 21: switch amino-acid numbering from absolute to relative.

For each ``{aa  position  interaction}`` tag, the absolute position is replaced
by the *n-th occurrence* of that amino acid in the protein sequence (so the
same residue always gets the same label regardless of where the sequence was
trimmed). Applied to both ``Fusion_SMILES`` (-> ``Fusion_SMILES_2``) and
``IntSeq_1`` (-> ``IntSeq_2``).

Output: ``bindingnet_pdbbind_intseqs_relative.csv``.
"""

import os
import sys
import re
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from tqdm import tqdm

_TAG = re.compile(r"\{([^}]+)\}")


def relative_numbering(seq, text):
    if pd.isna(text):
        return ""

    aa_counter = defaultdict(int)
    position_counts = {}
    for pos, aa in enumerate(seq, start=1):
        aa_counter[aa] += 1
        position_counts[pos] = aa_counter[aa]

    def replace(match):
        parts = match.group(1).split()
        if len(parts) != 3:
            return match.group(0)
        aa_type, pos_str, interaction = parts
        try:
            pos = int(pos_str)
        except ValueError:
            return match.group(0)
        if pos < 1 or pos > len(seq):
            return match.group(0)
        occurrence = position_counts.get(pos, 0)
        return f"{{{aa_type} {occurrence} {interaction}}}"

    return _TAG.sub(replace, str(text))


def main():
    df = pd.read_csv(config.ATOM_IDX_FILTERED)

    fusion2, intseq2 = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Relative numbering"):
        seq = str(row["Sequence"])
        fusion2.append(relative_numbering(seq, row["Fusion_SMILES"]))
        intseq2.append(relative_numbering(seq, row["IntSeq_1"]))

    df["Fusion_SMILES_2"] = fusion2
    df["IntSeq_2"] = intseq2
    df.to_csv(config.RELATIVE_NUMBERED, index=False)
    print("Applied relative numbering.")


if __name__ == "__main__":
    main()
