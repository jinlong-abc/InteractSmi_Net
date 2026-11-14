"""Step 04: append the protein sequence to the SMILES/affinity table.

Joins the UniProt accession from step 03 against a UniProt FASTA to add the
protein sequence. Output: ``pdbbind_with_seq.csv`` with columns
``pdbid, UniProt, Molecule SMILES, Assay Result, Sequence``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from Bio import SeqIO


def load_sequences(fasta_path):
    """Return {accession: sequence} from a UniProt FASTA file."""
    seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        accession = record.id.split("|")[1] if "|" in record.id else record.id
        seq[accession] = str(record.seq)
    return seq


def main():
    prot_seq = load_sequences(config.IDMAPPING_FASTA)
    df = pd.read_csv(config.SMILES_AFFINITY)

    missing = []
    sequences = []
    for uniprot in df["UniProt"]:
        if uniprot not in prot_seq:
            missing.append(uniprot)
            sequences.append("")
        else:
            sequences.append(prot_seq[uniprot])

    df["Sequence"] = sequences
    df = df[df["Sequence"] != ""]
    df.to_csv(config.WITH_SEQ, index=False)

    print(f"Wrote {len(df)} rows with sequences. "
          f"Missing UniProt ids dropped: {len(set(missing))}")


if __name__ == "__main__":
    main()
