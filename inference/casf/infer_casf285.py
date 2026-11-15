"""PDBBind CASF-2016 (285 complexes) inference — without-decoder model.

Runs the *without-decoder* affinity model (num_head=11, n_atom=1549) on the
285-complex CASF-2016 core set, producing RMSE ≈ 1.30.

The model code is bundled in `model/` and the graph builder in
`graph_builder.py` (copied from `Arch_Lab/without_decoder_apply/src/`).

All weights/dicts/data default to paths inside this directory; run with no
arguments from the repository root:

    python inference/casf/infer_casf285.py
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model.model import FusionCPI  # noqa: E402
from graph_builder import MolecularGraphBuilder  # noqa: E402


def build_params():
    # matches without_decoder_apply/src/test/Test_Config.py + the training log.
    return argparse.Namespace(
        fusion_type="bidirectional",
        final_predict_layer="OuterProductLinear",
        comp_dim=100, prot_dim=100, gat_dim=100, latent_dim=100,
        num_head=11, dropout=0.3, alpha=0.1, num_layers=4,
    )


def batch_pad(arr):
    N = max(a.shape[0] for a in arr)
    if arr[0].ndim == 1:
        out = np.zeros((len(arr), N), dtype=np.int64)
        mask = np.zeros((len(arr), N), dtype=np.float32)
        for i, a in enumerate(arr):
            n = a.shape[0]
            out[i, :n] = a + 1  # 0 = padding index
            mask[i, :n] = 1.0
        return out, mask
    out = np.zeros((len(arr), N, N), dtype=np.int64)
    for i, a in enumerate(arr):
        n = a.shape[0]
        out[i, :n, :n] = a
    return out, None


def process_embeddings(uid_list, embed_dict, device):
    embeds = [embed_dict[u]["embed"].float() for u in uid_list]
    masks = [embed_dict[u]["mask"].float() for u in uid_list]
    return torch.stack(embeds, 0).to(device), torch.stack(masks, 0).to(device)


def spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "checkpoints/without_decoder.pt"))
    ap.add_argument("--dict-dir", default=str(HERE / "dicts/bindingdb_pdbbind_bindingnet_all"))
    ap.add_argument("--esmc", default=str(HERE / "casf285_esmc.pkl"))
    ap.add_argument("--data", default=str(HERE / "data/casf2016_285.csv"), help="casf2016_285.csv")
    ap.add_argument("--output", default=str(HERE / "casf285_predictions.tsv"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 1) read the table (columns: ID, UniProt, Assay Result, pInteraction, Normal SMILES, ...)
    ids, uids, labels, smiles = [], [], [], []
    with open(args.data, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ids.append(row[0])
            uids.append(row[1])
            labels.append(float(row[3]))
            smiles.append(row[4])

    # 2) build molecular graphs (radius=1, integer dict)
    builder = MolecularGraphBuilder(
        load_dicts=MolecularGraphBuilder.load_dicts(prefix=args.dict_dir)
    )
    compounds, adjacencies, keep = [], [], []
    for i, s in enumerate(smiles):
        g = builder.mol_to_graph(s, radius=1)
        if g is not None:
            compounds.append(g["atoms"])
            adjacencies.append(g["adjacency"])
            keep.append(i)
    if len(keep) != len(smiles):
        print(f"WARNING: {len(smiles) - len(keep)} SMILES failed to parse; skipped")
    ids = [ids[i] for i in keep]
    uids = [uids[i] for i in keep]
    labels = [labels[i] for i in keep]

    # 3) embeddings
    with open(args.esmc, "rb") as f:
        embed_dict = pickle.load(f)
    missing = sorted({u for u in uids if u not in embed_dict})
    if missing:
        raise SystemExit(f"missing embeddings for {len(missing)} proteins: {missing}")

    # 4) model (without-decoder, num_head=11)
    n_atom = len(builder.fingerprint_dict)
    model = FusionCPI(n_atom, build_params()).to(device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck))
    model.load_state_dict(state)  # strict: the checkpoint must match exactly
    model.eval()
    print(f"loaded checkpoint ({len(state)} keys); n_atom={n_atom}; "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # 5) batched inference
    preds = []
    with torch.no_grad():
        for s in range(0, len(compounds), args.batch_size):
            e = min(s + args.batch_size, len(compounds))
            atoms, amask = batch_pad(compounds[s:e])
            adj, _ = batch_pad(adjacencies[s:e])
            prot, pmask = process_embeddings(uids[s:e], embed_dict, device)
            out = model.inference(
                torch.LongTensor(atoms).to(device),
                torch.FloatTensor(amask).to(device),
                torch.LongTensor(adj).to(device),
                prot, pmask, device,
            )
            preds.append(out.view(-1).float().cpu().numpy())

    preds = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.float32)

    # 6) save + metrics
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ID", "UniProt", "pInteraction", "prediction"])
        for i in range(len(ids)):
            w.writerow([ids[i], uids[i], labels_arr[i], float(preds[i])])
    print(f"wrote {len(ids)} rows -> {args.output}")

    rmse = float(np.sqrt(np.mean((preds - labels_arr) ** 2)))
    pearson = float(np.corrcoef(preds, labels_arr)[0, 1])
    print(f"RMSE={rmse:.4f}  Pearson={pearson:.4f}  Spearman={spearman(preds, labels_arr):.4f}")


if __name__ == "__main__":
    main()
