"""Step 23: drop redundant VDW tags and complexes with too few interactions.

For each residue, a ``VDW`` tag is removed when a stronger interaction
(``Hy/HB/Pi/PC/IB``) with the same residue is also present. Rows left with fewer
than 3 tags in either fusion column are filtered out.

Output: ``bindingnet_pdbbind_intseqs_vdw_filtered.csv``.
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from tqdm import tqdm

_TAG = re.compile(r"\{(\w+)\s+(\d+)\s+(VDW|Hy|HB|Pi|PC|IB)\}")
_STRONG = {"Hy", "HB", "Pi", "PC", "IB"}
MIN_INTERACTIONS = 3


def remove_vdw_conflicts(text):
    if pd.isna(text):
        return ""
    text = str(text)

    by_residue = {}
    for m in _TAG.finditer(text):
        aa, pos, inter = m.groups()
        by_residue.setdefault((aa, pos), []).append((inter, m.group(0)))

    to_remove = set()
    for (aa, pos), entries in by_residue.items():
        types = {inter for inter, _ in entries}
        if "VDW" in types and (types & _STRONG):
            for inter, tag in entries:
                if inter == "VDW":
                    to_remove.add(tag)

    for tag in to_remove:
        text = text.replace(tag, "")
    return text


def main():
    df = pd.read_csv(config.ONE2THREE)

    df["Fusion_SMILES"] = [remove_vdw_conflicts(v) for v in tqdm(df["Fusion_SMILES"], desc="VDW Fusion_SMILES")]
    df["Fusion_SMILES_2"] = [remove_vdw_conflicts(v) for v in tqdm(df["Fusion_SMILES_2"], desc="VDW Fusion_SMILES_2")]

    def tag_count(text):
        return len(_TAG.findall(str(text))) if not pd.isna(text) else 0

    keep = (df["Fusion_SMILES"].apply(tag_count) >= MIN_INTERACTIONS) & \
           (df["Fusion_SMILES_2"].apply(tag_count) >= MIN_INTERACTIONS)
    df = df[keep].reset_index(drop=True)

    df.to_csv(config.VDW_FILTERED, index=False)
    print(f"Kept {len(df)} rows after VDW-conflict removal.")


if __name__ == "__main__":
    main()
