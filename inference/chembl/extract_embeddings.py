"""Extract the embeddings of a few example proteins from the ESM-C source pickle.

Builds a small ``embeddings/`` (fp16 + offsets + ids) containing only the given
proteins, so the inference script can run the bundled ``example.tsv`` without
needing the full 9.4 GB embedding file.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SRC = sys.argv[1]          # uniprot_sequences_esmc_embed.pkl (variable-length)
OUT = Path(sys.argv[2])    # inference/embeddings
IDS = sys.argv[3].split(",")

MAX_LENGTH = 1500

print(f"loading {SRC} ...", flush=True)
with open(SRC, "rb") as f:
    source = pickle.load(f)
source = {str(k): v for k, v in source.items()}
print(f"loaded {len(source)} proteins", flush=True)

# report the value format of the first entry
first_key = next(iter(source))
first_val = source[first_key]
print(f"first key={first_key!r} value type={type(first_val).__name__}", flush=True)

arrays, ids, dim = [], [], None
for uid in IDS:
    if uid not in source:
        print(f"!! {uid} not found in source", flush=True)
        continue
    value = source[uid]
    if isinstance(value, dict):
        value = value.get("embed", value)
    if hasattr(value, "detach"):          # torch tensor
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        print(f"!! {uid}: unexpected shape {value.shape}", flush=True)
        continue
    value = value[:MAX_LENGTH]
    dim = dim or value.shape[1]
    arrays.append(value.astype(np.float16, copy=False))
    ids.append(uid)
    print(f"{uid}: {value.shape} dtype={value.dtype}", flush=True)

if not arrays:
    raise SystemExit("no example protein found")

OUT.mkdir(parents=True, exist_ok=True)
total = sum(a.shape[0] for a in arrays)
mmap = np.memmap(OUT / "protein_embeddings.fp16", dtype=np.float16,
                 mode="w+", shape=(total, dim))
offsets = [0]
cursor = 0
for a in arrays:
    n = a.shape[0]
    mmap[cursor:cursor + n] = a
    cursor += n
    offsets.append(cursor)
mmap.flush()
np.save(OUT / "protein_offsets.npy", np.asarray(offsets, dtype=np.int64))
(OUT / "protein_ids.json").write_text(json.dumps(ids), encoding="utf-8")
print(f"saved {len(ids)} proteins, dim={dim}, total residues={total}", flush=True)
