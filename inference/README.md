# InteractSmi affinity-only inference

This directory runs inference with the InteractSmi **affinity-only** checkpoint
(FusionCPI, without the fingerprint decoder) used in the Chemical Science
benchmark. It is self-contained and independent of the multitask training code
in `../src/`.

The reported benchmark result on the ChEMBL 2022+ temporal test set is
**RMSE 1.197 / MAE 0.920 / Pearson 0.534 / Spearman 0.518** (n=181,326).

## Files

| File | Purpose |
|---|---|
| `model.py` / `fusion_layers.py` / `final_predict_layer_archs.py` | affinity-only FusionCPI model |
| `graph_builder.py` | builds the BACPI-style molecular graph from SMILES |
| `fingerprint_dict.pkl` | maps fingerprint keys to model atom ids (1331 entries) |
| `best_model.pt` | the benchmark checkpoint weights (4.6 MB) |
| `infer.py` | inference entry point |
| `example.tsv` | a few demo rows (`smiles` + `uniprot_id`) |

## Zenodo download

Two kinds of data are kept out of git and hosted on Zenodo:

1. **Protein embeddings** (required for inference). Download these three files
   and place them in an `embeddings/` subdirectory:

   ```
   embeddings/
   ├── protein_embeddings.fp16   (9.4 GB)
   ├── protein_ids.json          (78 KB)
   └── protein_offsets.npy       (62 KB)
   ```

   The embeddings are a single memory-mapped FP16 array of shape
   `(total_residues, 1152)`; `protein_offsets.npy` gives each protein's start/end,
   and `protein_ids.json` maps a UniProt id to its offset index.

2. **Full train / validation / test datasets** (ChEMBL 26 temporal split:
   `train.tsv`, `validation_2021.tsv`, `test_2022_plus.tsv`, plus the quality-audit
   annotations). These are also on Zenodo and are only needed if you want to
   reproduce the full benchmark evaluation rather than run a few predictions.

## Usage

Single prediction:

```bash
python infer.py --smiles "CC(C)Cc1ccccc1" --protein P00720
```

Batch prediction from a TSV with `smiles` and `uniprot_id` columns:

```bash
python infer.py --tsv input.tsv --output predictions.tsv
```

### Example

Run the bundled `example.tsv` (requires the Zenodo protein embeddings first):

```bash
python infer.py --tsv example.tsv --output example_predictions.tsv
```

Expected output is a copy of `example.tsv` with an extra `prediction` column
(predicted pAffinity). If a UniProt id is missing from `protein_ids.json`, the
script reports which one — pick another id from that file in that case.

## Requirements

* `torch`, `numpy`, `pandas`, `rdkit` (see `../requirements.txt`).

The model uses FP16 protein embeddings and runs on GPU by default (falls back
to CPU).
