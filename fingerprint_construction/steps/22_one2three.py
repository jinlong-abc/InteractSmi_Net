"""Step 22: convert one-letter amino-acid tags to three-letter.

Rewrites every ``{aa  position  interaction}`` tag in the fusion columns
(``Fusion_SMILES``, ``Fusion_SMILES_2``, ``IntSeq_1``, ``IntSeq_2``) so the
amino-acid type is the three-letter code (uppercase).

Output: ``bindingnet_pdbbind_intseqs_one2three.csv``.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from tqdm import tqdm

from common.constants import AA1_TO_AA3

_TAG = re.compile(r"\{(\w+)\s+(\d+)\s+(VDW|Hy|HB|Pi|PC|IB)\}")


def one2three(text):
    if pd.isna(text):
        return ""

    def replace(match):
        aa = AA1_TO_AA3.get(match.group(1), match.group(1))
        return f"{{{aa} {match.group(2)} {match.group(3)}}}"

    return _TAG.sub(replace, str(text))


def main():
    df = pd.read_csv(config.RELATIVE_NUMBERED)

    for col in ("Fusion_SMILES", "Fusion_SMILES_2", "IntSeq_1", "IntSeq_2"):
        df[col] = [one2three(v) for v in tqdm(df[col], desc=f"one2three {col}")]

    df.to_csv(config.ONE2THREE, index=False)
    print("Converted amino-acid tags to three-letter.")


if __name__ == "__main__":
    main()
