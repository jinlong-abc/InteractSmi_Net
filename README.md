# InteractSmi
**Learning the Language of Protein–Ligand Interactions**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jinlong-abc/InteractSmi_Net/blob/master/InteractSmi_Net_Demo.ipynb)

**Note:** InteractSmi technology is patent-pending.  

InteractSmi is a sequence-based fingerprint for encoding protein–ligand interactions into a compact and interpretable representation. Built on InteractSmi, InteractSmi-Net enables efficient binding-affinity prediction and interaction-pattern inference, even when reference protein or ligand structures are unavailable.


## InteractSmi Fingerprint

InteractSmi encodes protein-ligand interactions into a compact, interpretable sequence. It captures essential biochemical and geometric information, providing a “grammar” for molecular interplay. This fingerprint can be inferred from either the complex structure or separate protein and ligand sequences.

<p align="center">
  <img src="image/InteractSmi.svg" width="600"/>
</p>

### Building the fingerprint

The [`fingerprint_construction/`](fingerprint_construction/) directory contains the pipeline that turns a raw PDBBind dataset into InteractSmi fingerprints (proLIF and closest-atoms interaction fingerprints, merged as `prolif_closest` and rendered as AtomIdx fusion sequences). See [`fingerprint_construction/README.md`](fingerprint_construction/README.md) for setup and usage.

---

## InteractSmi-Net Model Architecture

InteractSmi-Net is a sequence-based neural network built on the InteractSmi fingerprint. The model is designed for fast and accurate binding affinity prediction and interaction inference. Its architecture efficiently captures long-range dependencies and spatial interaction patterns while remaining interpretable.

<p align="center">
  <img src="image/InteractSmi_Net.svg" width="750"/>
</p>

---

## Environment Setup

Create the environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate interactsmi_env
````

### Test the Environment

Run the test command:

```bash
python -m test.Test_model_evalue > evalue_log.txt 2>&1
```

This will execute the test script and save logs to `evalue_log.txt`.

---

## Training

First build the InteractSmi fingerprints for your data with [`fingerprint_construction/`](fingerprint_construction/) (see above), then prepare the ESMC protein embeddings and the CSV/HDF5 inputs. Run the training from `src/`:

```bash
# Stage 1 -- fingerprint pretraining (no affinity supervision)
python -m model.train --stage 1 --affinity_weight 0.0 --sequence_weight 1.0 \
    -train_csv_data_path /path/to/train.csv -test_csv_data_path /path/to/test.csv \
    -train_dataset_path /path/to/train.h5 -test_dataset_path /path/to/test.h5 \
    -prot_esmc_embed /path/to/esmc.pkl -vocab /path/to/vocab.pkl

# Stage 2 -- fingerprint fine-tuning (PDBbind train only)
python -m model.train --stage 2 --affinity_weight 0.0 --sequence_weight 1.0 \
    --finetune_from <stage1>/test_best_models/best_sequence_loss.pt \
    ... (same data arguments)

# Stage 3 -- affinity prediction
# PDBbind: joint affinity + fingerprint
python -m model.train --stage 3 --task pdbbind --affinity_weight 1.0 --sequence_weight 0.1 \
    --finetune_from <stage2>/test_best_models/best_sequence_loss.pt ...

# ChEMBL: affinity only, frozen decoder
python -m model.train --stage 3 --task chembl --affinity_weight 1.0 --sequence_weight 0.0 \
    --finetune_from <stage2>/test_best_models/best_sequence_loss.pt \
    --freeze_layers '{"transformer_decoder": true, "generator": true, "embedwithpos": true}' ...
```

All paths and hyperparameters are command-line arguments — see [`src/config.py`](src/config.py) for the full list and defaults.

---

## Dependencies

* Python 3.9
* CUDA 12.4 required for PyTorch GPU support
* Main dependencies include:

  * `pytorch-cuda`
  * `torchvision`, `torchaudio`
  * `ipykernel`
  * `pymol-open-source`
  * `openbabel`
  * `rdkit`

Check the log file if there are any errors during execution.

---

## Datasets

The datasets used in this study are publicly accessible from the following sources:

* **PDBbind v2020**: Available at the [PDBbind database](https://www.pdbbind-plus.org.cn/download).  
* **BindingNet**: Deposited in Zenodo, including Crystal Templates, High-confidence, Moderate-confidence, and Low-confidence subsets. Access via [BindingNet on Zenodo](https://zenodo.org/records/11218293).  
* **BindingDB**: Available from the [BindingDB repository](https://www.bindingdb.org/rwd/bind/index.jsp).

---

## Features

* High-fidelity protein-ligand fingerprinting
* Sequence-based model, no need for reference structures
* Fast training and inference (~10x speedup compared to complex-based methods)
* Interpretable binding pattern predictions
* Supports reverse target fishing applications

---

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE file for details.
