"""Step 19: render the interaction sequence back into the SMILES (Fusion SMILES).

Strips the atom-map numbers from ``Mapped_SMILES`` to get a plain ``SMILES``,
then inserts ``{amino_acid  position  interaction}`` annotations directly after
the corresponding atom to produce ``Fusion_SMILES``.

Output: ``bindingnet_pdbbind_intseqs_fusion_smiles.csv`` (9 columns:
``..., SMILES, Fusion_SMILES, ...``).
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def de_map_smiles(mapped_smiles):
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=False, allHsExplicit=False)


def get_atom_positions(smiles):
    """Return (start, end) character spans of each atom token in ``smiles``."""
    atoms = []
    i, n = 0, len(smiles)
    while i < n:
        if smiles[i] == "[":
            close = smiles.find("]", i)
            if close == -1:
                break
            atoms.append((i, close + 1))
            i = close + 1
        elif i < n - 1 and smiles[i] in ("B", "C", "S") and smiles[i + 1] in ("r", "l", "i"):
            atoms.append((i, i + 2))
            i += 2
        elif smiles[i] in {"N", "O", "S", "P", "F", "I", "B", "C",
                           "n", "c", "o", "s", "p", "b"}:
            atoms.append((i, i + 1))
            i += 1
        else:
            i += 1
    return atoms


def build_fusion_smiles(mapped_smiles, intseq):
    output_smiles = de_map_smiles(mapped_smiles)
    if not output_smiles or Chem.MolFromSmiles(output_smiles) is None:
        return output_smiles, ""

    atoms = get_atom_positions(output_smiles)
    chars = list(output_smiles)

    annotated = []
    for match in re.findall(r"\{([^}]+)\}", intseq):
        parts = match.split()
        if len(parts) < 3:
            continue
        try:
            atom_index = int(parts[0])
        except ValueError:
            continue
        annotation = " ".join(parts[1:])
        annotated.append((atom_index, annotation))

    # insert from largest atom index to smallest so earlier offsets stay valid
    for atom_index, annotation in sorted(annotated, reverse=True):
        if 1 <= atom_index < len(atoms) + 1:
            _, end = atoms[atom_index - 1]
            chars.insert(end, "{" + annotation + "}")

    return output_smiles, "".join(chars)


def main():
    df = pd.read_csv(config.MW_FILTERED)

    smiles_list, fusion_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Fusion SMILES"):
        s, f = build_fusion_smiles(row["Mapped_SMILES"], str(row["IntSeq_1"]))
        smiles_list.append(s)
        fusion_list.append(f)

    df["SMILES"] = smiles_list
    df["Fusion_SMILES"] = fusion_list
    df = df[["ID", "UniProt", "Assay Result", "Normal_SMILES", "Mapped_SMILES",
             "SMILES", "Fusion_SMILES", "IntSeq_1", "Sequence"]]
    df.to_csv(config.FUSION_SMILES_CSV, index=False)
    print(f"Built Fusion SMILES for {len(df)} rows.")


if __name__ == "__main__":
    main()
