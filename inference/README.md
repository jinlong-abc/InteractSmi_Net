# Inference

Two independent inference paths live here:

| Directory | Task | Model | Reported result |
|---|---|---|---|
| [`chembl/`](chembl/) | ChEMBL 2022+ temporal benchmark | affinity-only FusionCPI (num_head=5) | RMSE 1.197 / Pearson 0.534 (n=181,326) |
| [`casf/`](casf/) | PDBBind CASF-2016 core set (285 complexes) | multitask FusionCPI (num_head=5) and without-decoder (num_head=11) | RMSE 1.242 / 1.298 |

See the README in each subdirectory for setup and usage.
