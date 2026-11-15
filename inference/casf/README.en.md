# CASF-2016 inference

## Run

### Multitask affinity checkpoint

```bash
python inference/casf/infer_multitask.py
```

### Multitask fingerprint checkpoint

```bash
python inference/casf/infer_multitask.py \
    --checkpoint checkpoints/checkpoint_fp/checkpoint.pt \
    --graph-dict inference/casf/dicts/bindingnet_pdbbind
```

### Without-decoder model

```bash
python inference/casf/infer_casf285.py
```

Each script writes `casf285_predictions.tsv` (`ID, UniProt, pInteraction,
prediction`) and prints RMSE / Pearson / Spearman.