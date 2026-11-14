"""Step 15: join the IntSeq table with protein sequence and affinity.

Adds the UniProt id, affinity and protein sequence (from step 04) to each
filtered IntSeq row, and renames the first column ``PDBID`` -> ``ID`` to match
the downstream (phase-2) convention.

Output: ``final_pdbbind_intseqs.csv`` with columns
``ID, UniProt, Assay Result, Normal_SMILES, Mapped_SMILES, IntSeq_1, Sequence``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd


def main():
    seq_df = pd.read_csv(config.WITH_SEQ)          # pdbid, UniProt, SMILES, affinity, Sequence
    intseq_df = pd.read_csv(config.INTSEQ_FILTERED)  # PDBID, Normal_SMILES, Mapped_SMILES, IntSeq_1

    lookup = seq_df.set_index("pdbid")

    uniprot, affinity, sequence = [], [], []
    for pdbid in intseq_df["PDBID"]:
        if pdbid in lookup.index:
            uniprot.append(lookup.loc[pdbid, "UniProt"])
            affinity.append(lookup.loc[pdbid, "Assay Result"])
            sequence.append(lookup.loc[pdbid, "Sequence"])
        else:
            uniprot.append(None)
            affinity.append(None)
            sequence.append(None)

    intseq_df["UniProt"] = uniprot
    intseq_df["Assay Result"] = affinity
    intseq_df["Sequence"] = sequence

    intseq_df = intseq_df.rename(columns={"PDBID": "ID"})
    intseq_df = intseq_df[
        ["ID", "UniProt", "Assay Result", "Normal_SMILES",
         "Mapped_SMILES", "IntSeq_1", "Sequence"]
    ]
    intseq_df.to_csv(config.FINAL_INTSEQ, index=False)
    print(f"Wrote {len(intseq_df)} final IntSeq rows.")


if __name__ == "__main__":
    main()
