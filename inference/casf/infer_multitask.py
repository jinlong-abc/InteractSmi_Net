"""PDBBind CASF-2016 (285 complexes) inference — multitask checkpoint.

Runs the repository's multitask FusionCPI checkpoint
(../../checkpoints/checkpoint_affinity/checkpoint1.pt, num_head=5) on the
285-complex CASF-2016 core set, producing RMSE ≈ 1.2 (Pearson ≈ 0.8).

All weights/dicts/data default to paths inside this directory; run with no
arguments from the repository root:

    python inference/casf/infer_multitask.py
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from model.model import FusionCPI  # noqa: E402
from data.graph_builder import MolecularGraphBuilder  # noqa: E402


def build_params():
    return argparse.Namespace(
        multitask=False,
        fusion_type="bidirectional",
        final_predict_layer="OuterProductLinear",
        comp_dim=100, prot_dim=100, gat_dim=100, latent_dim=100,
        num_head=5, dropout=0.3, alpha=0.1, num_layers=4,
    )


def batch_pad(arr):
    N = max(a.shape[0] for a in arr)
    if arr[0].ndim == 1:
        out = np.zeros((len(arr), N), dtype=np.int64)
        mask = np.zeros((len(arr), N), dtype=np.float32)
        for i, a in enumerate(arr):
            n = a.shape[0]
            out[i, :n] = a + 1
            mask[i, :n] = 1.0
        return out, mask
    out = np.zeros((len(arr), N, N), dtype=np.int64)
    for i, a in enumerate(arr):
        n = a.shape[0]
        out[i, :n, :n] = a
    return out, None


def spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints/checkpoint_affinity/checkpoint1.pt"))
    ap.add_argument("--graph-dict", default=str(HERE / "dicts/pdbbind_more_data"),
                    help="dir with the 733-key atom/bond/fingerprint dicts")
    ap.add_argument("--esmc", default=str(HERE / "casf285_esmc.pkl"))
    ap.add_argument("--data", default=str(HERE / "data/casf2016_285.csv"), help="casf2016_285.csv")
    ap.add_argument("--output", default=str(HERE / "casf285_predictions.tsv"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # graph builder uses the 733-key dict
    builder = MolecularGraphBuilder(
        load_dicts=MolecularGraphBuilder.load_dicts(prefix=args.graph_dict)
    )
    print(f"builder fingerprint keys: {len(builder.fingerprint_dict)}")

    ids, uids, labels, smiles = [], [], [], []
    with open(args.data, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            ids.append(row[0]); uids.append(row[1]); labels.append(float(row[3])); smiles.append(row[4])

    compounds, adjacencies, keep = [], [], []
    for i, s in enumerate(smiles):
        g = builder.mol_to_graph(s, radius=1)
        if g is not None:
            compounds.append(g["atoms"]); adjacencies.append(g["adjacency"]); keep.append(i)
    uids = [uids[i] for i in keep]; labels = [labels[i] for i in keep]

    with open(args.esmc, "rb") as f:
        embed_dict = pickle.load(f)
    missing = sorted({u for u in uids if u not in embed_dict})
    if missing:
        raise SystemExit(f"missing embeddings: {missing}")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck))
    n_atom = state["embedding_layer_atom.weight"].shape[0] - 1
    model = FusionCPI(n_atom, build_params()).to(device)
    model.load_state_dict(state, strict=False)  # skip decoder keys
    model.eval()
    print(f"n_atom (from checkpoint): {n_atom}")

    preds = []
    with torch.no_grad():
        for s in range(0, len(compounds), args.batch_size):
            e = min(s + args.batch_size, len(compounds))
            atoms, amask = batch_pad(compounds[s:e])
            adj, _ = batch_pad(adjacencies[s:e])
            prot = torch.stack([embed_dict[u]["embed"].float() for u in uids[s:e]], 0).to(device)
            pmask = torch.stack([embed_dict[u]["mask"].float() for u in uids[s:e]], 0).to(device)
            out = model.inference(torch.LongTensor(atoms).to(device),
                                  torch.FloatTensor(amask).to(device),
                                  torch.LongTensor(adj).to(device), prot, pmask)
            preds.append(out.view(-1).float().cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.array(labels, dtype=np.float32)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ID", "UniProt", "pInteraction", "prediction"])
        for i in range(len(uids)):
            w.writerow([ids[keep[i]], uids[i], labels[i], float(preds[i])])

    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    pearson = float(np.corrcoef(preds, labels)[0, 1])
    print(f"n={len(preds)}  RMSE={rmse:.4f}  Pearson={pearson:.4f}  Spearman={spearman(preds, labels):.4f}")


if __name__ == "__main__":
    main()
