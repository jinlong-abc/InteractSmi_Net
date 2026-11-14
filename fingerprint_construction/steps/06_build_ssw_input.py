"""Step 06: build the input FASTA files for the SSW sequence alignment.

Writes two FASTA files with one entry per (pdbid, chain):

* ``pdbbind_ssw_target.fasta`` -- the PDB chain sequence (one-letter)
* ``pdbbind_ssw_query.fasta``  -- the matching UniProt sequence

The external ``ssw`` tool then aligns these into ``pdbbind_seq_align.txt``
(see README), which later steps read to map PDB residue numbers onto the
UniProt sequence.
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from common.constants import AA3_TO_AA1


def convert_to_single_letter(three_letter_seq):
    return "".join(
        AA3_TO_AA1.get(aa, "X") for aa in three_letter_seq.split("-") if aa
    )


def load_fasta_dict(fasta_path):
    seq = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        accession = record.id.split("|")[1] if "|" in record.id else record.id
        seq[accession] = str(record.seq)
    return seq


def main():
    with open(config.ALL_PDBS_PKL, "rb") as f:
        data_class_dict = pickle.load(f)

    # pdbid -> canonical UniProt accession.
    pdbid_to_uniprot_raw = dict(pd.read_csv(config.PDBID_UNIPROT_MAP).values)
    from_to_entry = dict(
        zip(pd.read_csv(config.IDMAPPING_TSV, sep="\t")["From"],
            pd.read_csv(config.IDMAPPING_TSV, sep="\t")["Entry"])
    )
    fasta_dict = load_fasta_dict(config.IDMAPPING_FASTA)

    missing_map, missing_seq = set(), set()
    n_pairs = 0

    with open(config.SSW_TARGET_FASTA, "w") as target_f, \
         open(config.SSW_QUERY_FASTA, "w") as query_f:

        for pdbid in tqdm(data_class_dict, desc="Building SSW input"):
            if pdbid not in pdbid_to_uniprot_raw:
                missing_map.add(pdbid)
                continue
            uniprot = from_to_entry.get(pdbid_to_uniprot_raw[pdbid])
            if uniprot not in fasta_dict:
                missing_seq.add(uniprot)
                continue
            query_seq = fasta_dict[uniprot]

            protein_sequences = data_class_dict[pdbid].get_protein_sequences()
            for chain_id, three_letter in protein_sequences.items():
                single = convert_to_single_letter(three_letter)
                target_f.write(f">{pdbid}_{chain_id}\n{single}\n")
                query_f.write(f">{uniprot}\n{query_seq}\n")
                n_pairs += 1

    print(f"Wrote {n_pairs} target/query pairs.")
    if missing_map:
        print(f"  {len(missing_map)} pdbids missing a UniProt mapping.")
    if missing_seq:
        print(f"  {len(missing_seq)} UniProt ids missing a sequence.")


if __name__ == "__main__":
    main()
