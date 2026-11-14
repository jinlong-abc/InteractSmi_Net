"""Step 02: extract the PDB-id -> UniProt-id mapping from the PDBBind index.

Reads ``INDEX_general_PL_name.2020`` (columns: pdbid, year, uniprot, name) and
writes two files:

* ``pdbid_uniprotid_map.csv`` -- ``pdbid,uniprot`` pairs
* ``uniprotid.csv``           -- the set of distinct UniProt ids
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    if not os.path.exists(config.INDEX_PL_NAME):
        raise FileNotFoundError(
            f"INDEX file not found: {config.INDEX_PL_NAME}. "
            f"Check config.PDBBIND_DIR ({config.PDBBIND_DIR})."
        )

    uniprot_ids = set()
    rows = []

    with open(config.INDEX_PL_NAME) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 3:
                continue
            pdbid, uniprot = cols[0], cols[2]
            rows.append((pdbid, uniprot))
            uniprot_ids.add(uniprot)

    with open(config.PDBID_UNIPROT_MAP, "w") as f:
        f.write("pdbid,uniprot\n")
        for pdbid, uniprot in rows:
            f.write(f"{pdbid},{uniprot}\n")

    with open(config.UNIPROT_LIST, "w") as f:
        f.write("uniprot\n")
        for uniprot in sorted(uniprot_ids):
            f.write(f"{uniprot}\n")

    print(f"Wrote {len(rows)} pdbid->uniprot pairs "
          f"({len(uniprot_ids)} distinct UniProt ids).")


if __name__ == "__main__":
    main()
