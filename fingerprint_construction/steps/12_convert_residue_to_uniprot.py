"""Step 12: convert PDB residue numbers in the IntSeq to UniProt positions.

For each ``{atom  pdb_residue  interaction}`` token produced by step 11, the
PDB residue number is translated to its 1-based UniProt position using the
interaction pickles (step 08/09). Closest-atoms entries (whose third field is a
distance) are relabelled ``VDW``.

Output: ``pdbbind_mapped_smiles_with_intseq_2.csv``.
"""

import os
import sys
import pickle
import csv
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_TOKEN = re.compile(r"\{(\d+)\s+(\d+)\s+([\w.]+)\}")


def main():
    with open(config.PROLIF_MAPPED_PKL, "rb") as f:
        prolif = {i.chemblid_all: i for i in pickle.load(f)}
    with open(config.CLOSEST_PKL, "rb") as f:
        closest = {i.chemblid_all: i for i in pickle.load(f)}

    def convert(pdbid, match):
        atom_num, pdb_res_num, inter = match.group(1), int(match.group(2)), match.group(3)
        if inter.isalpha():  # proLIF type -> map via prolif record
            p = prolif.get(pdbid)
            if p is None or pdb_res_num not in p.protein_residues:
                return match.group(0)
            idx = p.protein_residues.index(pdb_res_num)
            if idx >= len(p.uniprot_idx_list):
                return match.group(0)
            uniprot_res = p.uniprot_idx_list[idx] + 1
            return f"{{{atom_num} {uniprot_res} {inter}}}"
        else:  # numeric distance -> closest-atoms record, relabel VDW
            c = closest.get(pdbid)
            if c is None or pdb_res_num not in c.interact_residue_pdb_ids:
                return match.group(0)
            idx = c.interact_residue_pdb_ids.index(pdb_res_num)
            if idx >= len(c.uniprot_idx_list):
                return match.group(0)
            uniprot_res = c.uniprot_idx_list[idx] + 1
            return f"{{{atom_num} {uniprot_res} VDW}}"

    with open(config.INTSEQ_CSV, newline="") as infile, \
         open(config.INTSEQ_UNIPROT_CSV, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            pdbid = row["PDBID"]
            row["IntSeq_1"] = _TOKEN.sub(lambda m: convert(pdbid, m), row["IntSeq_1"])
            writer.writerow(row)

    print("Converted IntSeq residue numbers to UniProt positions.")


if __name__ == "__main__":
    main()
