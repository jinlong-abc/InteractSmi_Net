"""Extract the CASF-2016 proteins' ESMC embeddings from the big PDBBind cache.

One-time helper: loads pdbbind_bindingnet_esmc_pad_mask1024.pkl (17.5 GB) and
writes only the proteins that appear in the CASF-2016 table, so inference does
not have to re-read the full file each time.

Run from the repo root:

    python inference/extract_casf285_embeddings.py \
        --big /path/to/pdbbind_bindingnet_esmc_pad_mask1024.pkl \
        --csv /path/to/casf2016_285.csv \
        --out inference/casf285_esmc.pkl
"""

import argparse
import csv
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True,
                    help="pdbbind_bindingnet_esmc_pad_mask1024.pkl (17.5 GB)")
    ap.add_argument("--csv", required=True,
                    help="casf2016_285.csv")
    ap.add_argument("--out", default="casf285_esmc.pkl")
    args = ap.parse_args()

    uids = []
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row[1] not in uids:
                uids.append(row[1])
    print(f"{len(uids)} unique UniProt IDs in the CASF-2016 table")

    print("loading the big embedding pickle (17.5 GB, a minute or two)...")
    with open(args.big, "rb") as f:
        big = pickle.load(f)
    print(f"big pickle holds {len(big)} proteins")

    out, missing = {}, []
    for uid in uids:
        if uid in big:
            out[uid] = big[uid]
        else:
            missing.append(uid)

    print(f"extracted {len(out)}/{len(uids)}; missing {len(missing)}")
    if missing:
        print("MISSING:", missing)

    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
