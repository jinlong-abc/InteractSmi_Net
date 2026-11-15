"""Affinity-only inference for the InteractSmi benchmark checkpoint.

Loads the FusionCPI affinity-only model and its weights, builds molecular
graphs on the fly from SMILES (BACPI-style fingerprint, via the bundled
``fingerprint_dict.pkl``), reads protein ESMC embeddings from the
Zenodo-downloaded memory-mapped files, and predicts pAffinity.

Quick start
-----------
1. Download the Zenodo protein-embedding files into ``embeddings/``:
   ``protein_embeddings.fp16``, ``protein_ids.json``, ``protein_offsets.npy``.
2. Single prediction::

       python infer.py --smiles "CC(C)Cc1ccccc1" --protein P00720

3. Batch prediction from a TSV (columns ``smiles`` and ``uniprot_id``)::

       python infer.py --tsv input.tsv --output predictions.tsv
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from model import FusionCPI
from graph_builder import build_graph

HERE = Path(__file__).resolve().parent
EMBED_DIR = HERE / "embeddings"        # Zenodo files go here
WEIGHTS = HERE / "best_model.pt"
FP_DICT = HERE / "fingerprint_dict.pkl"

# Fixed architecture of the benchmark checkpoint (see its config.json).
MODEL_HP = dict(
    comp_dim=100, prot_dim=100, gat_dim=100, num_head=5,
    dropout=0.3, alpha=0.1, latent_dim=100, num_layers=4,
    fusion_type="bidirectional", final_predict_layer="OuterProductLinear",
)


class ProteinEmbeddingStore:
    """Reads the memory-mapped ESMC protein embeddings from the Zenodo files."""

    def __init__(self, embed_dir, dim=1152):
        embed_dir = Path(embed_dir)
        self.dim = dim
        ids = json.loads((embed_dir / "protein_ids.json").read_text())
        if isinstance(ids, dict):
            self.id_to_idx = {str(k): int(v) for k, v in ids.items()}
        else:
            self.id_to_idx = {str(v): i for i, v in enumerate(ids)}
        self.offsets = np.load(embed_dir / "protein_offsets.npy", mmap_mode="r")
        self.embeds = np.memmap(
            embed_dir / "protein_embeddings.fp16", dtype=np.float16, mode="r"
        ).reshape(-1, dim)

    def get(self, protein_id):
        idx = self.id_to_idx[str(protein_id)]
        s, e = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return np.array(self.embeds[s:e], dtype=np.float16)  # writable copy


def load_model(weights_path, n_atom, device):
    cfg = argparse.Namespace(**MODEL_HP)
    model = FusionCPI(n_atom, cfg).to(device)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


def predict(model, fp_dict, store, items, device, batch_size=32):
    """items: list of (smiles, protein_id). Batched to bound memory usage."""
    all_preds = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        graphs = [build_graph(smiles, fp_dict) for smiles, _ in batch]
        embeds = [store.get(pid) for _, pid in batch]

        n = len(batch)
        max_m = max(len(g[0]) for g in graphs)
        max_p = max(len(e) for e in embeds)
        dim = embeds[0].shape[1]

        atoms = torch.zeros((n, max_m), dtype=torch.long)
        amask = torch.zeros((n, max_m), dtype=torch.bool)
        adj = torch.zeros((n, max_m, max_m), dtype=torch.bool)
        prot = torch.zeros((n, max_p, dim), dtype=torch.float32)
        pmask = torch.zeros((n, max_p), dtype=torch.bool)

        for j, ((ids, edges), emb) in enumerate(zip(graphs, embeds)):
            m, p = len(ids), len(emb)
            atoms[j, :m] = torch.from_numpy(ids) + 1
            amask[j, :m] = True
            adj[j, edges[:, 0], edges[:, 1]] = True
            prot[j, :p] = torch.from_numpy(emb).float()
            pmask[j, :p] = True

        with torch.no_grad():
            out = model(atoms.to(device), amask.to(device), adj.to(device),
                        prot.to(device), pmask.to(device))
        all_preds.append(out.view(-1).float().cpu().numpy())

    return np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", help="single SMILES for --protein")
    ap.add_argument("--protein", help="single UniProt id for --smiles")
    ap.add_argument("--tsv", help="input TSV with `smiles` and `uniprot_id` columns")
    ap.add_argument("--output", default="predictions.tsv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--embed-dir", default=str(EMBED_DIR))
    ap.add_argument("--batch-size", type=int, default=32, help="inference batch size")
    args = ap.parse_args()

    with open(FP_DICT, "rb") as f:
        fp_dict = pickle.load(f)
    store = ProteinEmbeddingStore(args.embed_dir)
    device = torch.device(args.device)
    model = load_model(WEIGHTS, len(fp_dict), device)

    if args.tsv:
        import pandas as pd
        df = pd.read_csv(args.tsv, sep="\t")
        smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
        prot_col = ("uniprot_id" if "uniprot_id" in df.columns
                    else "protein" if "protein" in df.columns else df.columns[1])
        items = list(zip(df[smiles_col].astype(str), df[prot_col].astype(str)))
        preds = predict(model, fp_dict, store, items, device, batch_size=args.batch_size)
        df["prediction"] = preds
        df.to_csv(args.output, sep="\t", index=False)
        print(f"wrote {len(df)} predictions to {args.output}")
    else:
        if not (args.smiles and args.protein):
            raise SystemExit("provide --smiles and --protein, or --tsv")
        pred = predict(model, fp_dict, store, [(args.smiles, args.protein)], device,
                       batch_size=1)
        print(f"predicted pAffinity: {pred[0]:.4f}")


if __name__ == "__main__":
    main()
