"""Step 03: build the SMILES + affinity table from the PDBBind index.

For each complex in ``INDEX_general_PL.2020`` this script

* parses the affinity (``Kd=...``/``Ki=...``/``IC50=...``) and normalises it to nM,
* obtains the ligand SMILES from the prepared ligand PDB (step 01),
* maps the PDB id to a UniProt accession via step 02 + the idmapping table.

Output: ``pdbbind_smiles_affinity.csv`` with columns
``pdbid, UniProt, Molecule SMILES, Assay Result``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from rdkit import Chem


def load_idmapping(tsv_path):
    """Load a two-column ``From``/``Entry`` idmapping table into a dict."""
    df = pd.read_csv(tsv_path, sep="\t")
    return dict(zip(df["From"], df["Entry"]))


def parse_affinity(raw):
    """Normalise an INDEX affinity field (e.g. ``Kd=49uM``) to ``Kd = 49000.0 nM``.

    Returns ``None`` for qualitative entries (``<``, ``>``, ``~``).
    """
    raw = raw.split("//")[0].strip()
    if any(c in raw for c in ("<", ">", "~")):
        return None
    if "=" not in raw:
        return None
    try:
        measure = raw.split("=")[0].strip()
        value = float(raw.split("=")[1][:-2])
        unit = raw.split("=")[1][-2:]
        if unit == "uM":
            value, unit = value * 1000, "nM"
        elif unit == "mM":
            value, unit = value * 1_000_000, "nM"
        elif unit == "pM":
            value, unit = value / 1000, "nM"
        elif unit == "fM":
            value, unit = value / 1_000_000, "nM"
        return f"{measure} = {value} {unit}"
    except (ValueError, IndexError):
        return None


def main():
    if not os.path.exists(config.INDEX_PL):
        raise FileNotFoundError(f"INDEX not found: {config.INDEX_PL}")

    # pdbid -> canonical UniProt accession (step 02 + idmapping).
    pdbid_to_uniprot_raw = dict(
        pd.read_csv(config.PDBID_UNIPROT_MAP).values
    )
    from_to_entry = load_idmapping(config.IDMAPPING_TSV)

    rows = []
    skipped = 0

    with open(config.INDEX_PL) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("/")[0].strip().split()
            if len(cols) < 4:
                continue
            pdbid, affinity_raw = cols[0], cols[3]

            affinity = parse_affinity(cols[3])
            if affinity is None:
                skipped += 1
                continue

            ligand_pdb = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_ligand.pdb")
            if not os.path.exists(ligand_pdb):
                skipped += 1
                continue
            try:
                mol = Chem.MolFromPDBFile(ligand_pdb)
                if mol is None:
                    skipped += 1
                    continue
                smiles = Chem.MolToSmiles(mol)
            except Exception:
                skipped += 1
                continue

            if pdbid not in pdbid_to_uniprot_raw:
                skipped += 1
                continue
            uniprot = from_to_entry.get(pdbid_to_uniprot_raw[pdbid])
            if uniprot is None:
                skipped += 1
                continue

            rows.append((pdbid, uniprot, smiles, affinity))

    with open(config.SMILES_AFFINITY, "w") as f:
        f.write("pdbid,UniProt,Molecule SMILES,Assay Result\n")
        for pdbid, uniprot, smiles, affinity in rows:
            f.write(f"{pdbid},{uniprot},{smiles},{affinity}\n")

    print(f"Wrote {len(rows)} rows; skipped {skipped}.")


if __name__ == "__main__":
    main()
