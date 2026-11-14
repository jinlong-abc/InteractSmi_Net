"""Step 13: add the amino-acid type and abbreviate interaction types.

Expands each ``{atom  uniprot_pos  interaction}`` token from step 12 into
``{atom  amino_acid  uniprot_pos  abbreviated_interaction}``, using the protein
sequence to look up the residue and the interaction dictionary to abbreviate
proLIF type names (``Hydrophobic -> Hy``, ``HBDonor/HBAcceptor -> HB``, ...).

Output: ``pdbbind_mapped_smiles_with_intseq_3.csv``.
"""

import os
import sys
import csv
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from common.constants import INTERACTION_ABBREV

_TOKEN = re.compile(r"\{(\d+)\s+(\d+)\s+([\w.]+)\}")


def main():
    # pdbid -> protein sequence (as a list of one-letter residues)
    sequence_by_pdbid = {}
    with open(config.WITH_SEQ, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] != "pdbid":
                sequence_by_pdbid[row[0]] = list(row[4])

    with open(config.INTSEQ_UNIPROT_CSV, newline="") as infile, \
         open(config.INTSEQ_AA_CSV, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            pdbid = row["PDBID"]
            seq = sequence_by_pdbid.get(pdbid, [])

            def replace(match, seq=seq):
                atom_num = match.group(1)
                res_num = int(match.group(2))
                interaction = INTERACTION_ABBREV.get(match.group(3), match.group(3))
                if 0 < res_num <= len(seq):
                    aa = seq[res_num - 1]
                    return f"{{{atom_num} {aa} {res_num} {interaction}}}"
                return f"{{{atom_num} {res_num} {interaction}}}"

            row["IntSeq_1"] = _TOKEN.sub(replace, row["IntSeq_1"])
            writer.writerow(row)

    print("Added amino-acid types and abbreviated interaction names.")


if __name__ == "__main__":
    main()
