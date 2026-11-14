"""Step 14: drop invalid IntSeq tokens and complexes with too few interactions.

Removes ``{atom  aa  pos  interaction}`` tokens whose residue position is
outside ``[1, 1024]`` (missing SSW mapping produced ``-1``/``0``, and very long
chains are clipped), then drops any complex left with fewer than 3 tokens.

Output: ``pdbbind_intseqs_filtered.csv``.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd

# {atom  aa  pos  interaction}  -- valid, 4-part token
_TOKEN = re.compile(r"\{(\d+)\s([A-Za-z])\s(\d+)\s([A-Za-z]+)\}")

MIN_INTERACTIONS = 3
MAX_RESIDUE = 1024


def clean_intseq(text):
    if pd.isna(text):
        return ""
    kept = []
    for m in _TOKEN.finditer(str(text)):
        pos = int(m.group(3))
        if 1 <= pos <= MAX_RESIDUE:
            kept.append(m.group(0))
    return " ".join(kept)


def main():
    df = pd.read_csv(config.INTSEQ_AA_CSV)
    df["IntSeq_1"] = df["IntSeq_1"].apply(clean_intseq)

    # keep only rows with at least MIN_INTERACTIONS valid tokens
    counts = df["IntSeq_1"].apply(lambda s: len(_TOKEN.findall(s)))
    df = df[counts >= MIN_INTERACTIONS].reset_index(drop=True)

    df.to_csv(config.INTSEQ_FILTERED, index=False)
    print(f"Kept {len(df)} complexes after filtering.")


if __name__ == "__main__":
    main()
