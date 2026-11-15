"""InteractSmi-Net three-stage training script.

Implements the training protocol described in the paper:

* Stage 1 -- interaction-fingerprint pretraining (BindingNet + PDBbind train,
  no affinity supervision).
* Stage 2 -- fingerprint-generation fine-tuning (PDBbind train only).
* Stage 3 -- downstream affinity prediction (PDBBind: joint affinity + fingerprint
  loss; ChEMBL: affinity loss only with the fingerprint decoder frozen).

The stage is selected by ``--stage``; the actual objective is driven by
``-affinity_weight`` / ``-sequence_weight``, and finetuning by
``-finetune_from`` / ``-finetune_lr`` / ``-freeze_layers``. See config.py for
the exact command lines.

Run from ``src/``:

    python -m model.train --stage 1 ...

"""

import os
import json
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model.model import FusionCPI
from model.decoder_utils import FusionCPILoss, AdaptiveFusionCPILoss
from data.data_utils import prepare_datasets, load_protein_embeddings
from config import get_config
from model.train_utils import (
    train_one_epoch,
    save_checkpoint,
    evaluate_model,
    save_best_model,
)


def setup_environment(params):
    if params.mode == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = params.cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA unavailable; falling back to CPU.")
    else:
        device = torch.device("cpu")
    print(f"The code runs on: {device}")
    return device


def setup_model_and_optimizer(atom_dict_len, params, device, vocab):
    """Build the multitask model, the joint loss, optimizer and scheduler."""
    model = FusionCPI(atom_dict_len, params, vocab)
    model.to(device)

    criterion = FusionCPILoss(
        affinity_weight=params.affinity_weight,
        sequence_weight=params.sequence_weight,
        ignore_index=-100,
    )
    print(
        f"Joint loss - affinity_weight: {params.affinity_weight}, "
        f"sequence_weight: {params.sequence_weight}"
    )

    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return model, criterion, optimizer, scheduler


def freeze_model_layers(model, freeze_config):
    """Freeze parameters according to a dict.

    ``freeze_config`` supports two kinds of entries:
      * ``"decoder_layers": [0, 1, ...]`` -- freeze specific decoder layers
      * ``"<substring>": true`` -- freeze every parameter whose name contains the
        substring (e.g. ``"transformer_decoder"``, ``"generator"``,
        ``"embedwithpos"``).
    """
    frozen, total = 0, 0
    for name, param in model.named_parameters():
        total += 1

        if "decoder_layers" in freeze_config and name.startswith("transformer_decoder.layers."):
            layer_id = int(name.split(".")[2])
            if layer_id in freeze_config["decoder_layers"]:
                param.requires_grad = False
                frozen += 1
                print(f"Frozen decoder layer {layer_id}: {name}")
            continue

        for module_name, freeze_flag in freeze_config.items():
            if module_name == "decoder_layers":
                continue
            if freeze_flag is True and module_name in name:
                param.requires_grad = False
                frozen += 1
                print(f"Frozen module {module_name}: {name}")

    print(f"[Freeze] {frozen}/{total} parameters frozen ({100.0 * frozen / total:.2f}%)")


def train_model(model, criterion, optimizer, scheduler, train_data, dev_data, test_data,
                device, params, uniprotid_prot_embed_dict, vocab):
    """Run the training loop (joint multitask objective)."""
    print("Starting training...")

    global_step = 0
    idx = np.arange(len(train_data[0]))
    best_dev = {"loss": float("inf"), "affinity_loss": float("inf"),
                "sequence_loss": float("inf"), "rmse": float("inf")}
    best_test = dict(best_dev)

    os.makedirs(params.checkpoint_path, exist_ok=True)
    dev_best_dir = os.path.join(params.checkpoint_path, params.name, "dev_best_models")
    test_best_dir = os.path.join(params.checkpoint_path, params.name, "test_best_models")
    os.makedirs(dev_best_dir, exist_ok=True)
    os.makedirs(test_best_dir, exist_ok=True)

    for epoch in range(params.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{params.num_epochs} ===")
        global_step = train_one_epoch(
            global_step, model, criterion, optimizer, idx, train_data, device,
            params, uniprotid_prot_embed_dict, vocab=vocab, multitask=True,
        )
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        dev_metrics, test_metrics = evaluate_model(
            model, dev_data, test_data, params.batch_size, device,
            criterion, uniprotid_prot_embed_dict, vocab=vocab, params=params,
        )
        (rmse_dev, _, _, loss_dev, aff_dev, seq_dev) = dev_metrics
        (rmse_test, _, _, loss_test, aff_test, seq_test) = test_metrics

        if (epoch + 1) % params.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch + 1, global_step,
                            params.checkpoint_path, params.name)

        metrics_dev = {"loss": loss_dev, "affinity_loss": aff_dev,
                       "sequence_loss": seq_dev, "rmse": rmse_dev}
        metrics_test = {"loss": loss_test, "affinity_loss": aff_test,
                        "sequence_loss": seq_test, "rmse": rmse_test}

        best_dev = save_best_model(model, optimizer, epoch + 1, global_step,
                                   metrics_dev, best_dev, save_dir=dev_best_dir,
                                   multitask=True)
        best_test = save_best_model(model, optimizer, epoch + 1, global_step,
                                    metrics_test, best_test, save_dir=test_best_dir,
                                    multitask=True)

        print(f"Best Dev RMSE so far: {best_dev['rmse']:.4f}")
        print(f"Best Test RMSE so far: {best_test['rmse']:.4f}")

    return best_dev, best_test


def main():
    params = get_config()
    params.multitask = True  # the three-stage protocol always uses the multitask model
    print("Parameters:", params)
    print(f"Stage: {params.stage}, task: {params.task}")

    device = setup_environment(params)

    # Data / embeddings / vocab
    train_data, dev_data, test_data, atom_dict_len = prepare_datasets(params)
    uniprotid_prot_embed_dict = load_protein_embeddings(params)
    with open(params.vocab, "rb") as f:
        vocab = pickle.load(f)

    # Model + loss + optimizer
    model, criterion, _, _ = setup_model_and_optimizer(
        atom_dict_len, params, device, vocab
    )

    # Finetuning: load a pretrained checkpoint, optionally freeze layers,
    # and re-create the optimizer over trainable parameters only.
    if params.finetune_from:
        if os.path.exists(params.finetune_from):
            checkpoint = torch.load(params.finetune_from, map_location=device)
            model.load_state_dict(checkpoint["model_state"], strict=False)
            print(f"Loaded pretrained model from {params.finetune_from}")

            if params.freeze_layers:
                freeze_config = json.loads(params.freeze_layers)
                freeze_model_layers(model, freeze_config)
        else:
            print(f"Pretrained model not found: {params.finetune_from}")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params.finetune_lr,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        print(f"Finetuning with lr={params.finetune_lr}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, amsgrad=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)

    best_dev, best_test = train_model(
        model, criterion, optimizer, scheduler, train_data, dev_data, test_data,
        device, params, uniprotid_prot_embed_dict, vocab,
    )

    print("\n=== Training Completed ===")
    print(f"Best Dev RMSE: {best_dev['rmse']:.4f}")
    print(f"Best Test RMSE: {best_test['rmse']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
