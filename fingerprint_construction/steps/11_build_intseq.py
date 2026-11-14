"""Step 11: build the interaction sequence (IntSeq) with the prolif_closest merge.

Combines the proLIF (step 09) and closest-atoms (step 08) interactions: where
proLIF already covers a residue, the closest entry for that residue is dropped;
the two are otherwise merged and de-duplicated. Each resulting interaction is
emitted as ``{ligand_atom  residue_pdb_id  interaction}``.

Output: ``pdbbind_mapped_smiles_with_intseq.csv`` (step 10 columns + ``IntSeq_1``).
"""

import os
import sys
import pickle
import csv
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_interactions(pkl_path):
    with open(pkl_path, "rb") as f:
        interactions = pickle.load(f)
    return {i.chemblid_all: i for i in interactions}


def main():
    prolif = load_interactions(config.PROLIF_MAPPED_PKL)
    closest = load_interactions(config.CLOSEST_PKL)

    intseq_by_pdbid = {}
    for pdbid, c in closest.items():
        # closest map: {ligand_atom: [(residue_pdb_id, interaction)]}
        map_closest = defaultdict(list)
        for num, amino, inter in zip(
            c.interact_ligand_atom_serial_numbers,
            c.interact_residue_pdb_ids,
            c.distances,
        ):
            map_closest[num + 1].append((amino, inter))

        if pdbid not in prolif:
            merged = map_closest
        else:
            p = prolif[pdbid]
            map_prolif = defaultdict(list)
            for num, amino, inter in zip(
                p.ligand_indices, p.protein_residues, p.interaction_types
            ):
                map_prolif[num + 1].append((amino, inter))

            # drop closest entries whose residue proLIF already covers
            aminos_in_prolif = {a for vals in map_prolif.values() for a, _ in vals}
            for k in list(map_closest):
                map_closest[k] = [i for i in map_closest[k] if i[0] not in aminos_in_prolif]

            # merge, de-duplicating by residue
            for k in map_closest:
                if k not in map_prolif:
                    map_prolif[k] = map_closest[k]
                else:
                    seen = {a for a, _ in map_prolif[k]}
                    for item in map_closest[k]:
                        if item[0] not in seen:
                            map_prolif[k].append(item)
                            seen.add(item[0])
            merged = map_prolif

        tokens = []
        for num in merged:
            for amino, inter in merged[num]:
                tokens.append(f"{{{num} {amino} {inter}}}")
        intseq_by_pdbid[pdbid] = " ".join(tokens)

    with open(config.MAPPED_SMILES, newline="") as infile, \
         open(config.INTSEQ_CSV, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["IntSeq_1"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            row["IntSeq_1"] = intseq_by_pdbid.get(row["PDBID"], "")
            writer.writerow(row)

    print(f"Built IntSeq_1 for {len(intseq_by_pdbid)} complexes.")


if __name__ == "__main__":
    main()
