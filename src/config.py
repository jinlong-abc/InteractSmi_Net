"""Unified configuration for the InteractSmi-Net three-stage training.

The model is trained in three stages (see the paper / README):

1. **Interaction-fingerprint pretraining** on the filtered BindingNet corpus +
   the PDBbind training subset, *without* affinity supervision.
   ``--affinity_weight 0.0 --sequence_weight 1.0``

2. **Fingerprint-generation fine-tuning** on the PDBbind training subset only.
   ``--affinity_weight 0.0 --sequence_weight 1.0 --finetune_from <stage1.pt>``

3. **Affinity prediction** initialized from the stage-2 checkpoint:
   * PDBbind -- affinity + fingerprint generation optimized jointly
     ``--affinity_weight 1.0 --sequence_weight 0.1 --finetune_from <stage2.pt>``
   * ChEMBL  -- affinity only, fingerprint decoder frozen
     ``--task chembl --affinity_weight 1.0 --sequence_weight 0.0
       --finetune_from <stage2.pt> --freeze_layers '{"transformer_decoder": true,
       "generator": true, "embedwithpos": true}'``

Every path defaults to a placeholder; set it to your own copy (or export the
corresponding environment-independent value directly on the command line).
"""

import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="InteractSmi-Net three-stage training"
    )

    # ---------------- training stage / downstream task ----------------
    parser.add_argument("--stage", type=str, default="1", choices=["1", "2", "3"],
                        help="1=fingerprint pretraining, 2=fingerprint finetune, "
                             "3=affinity prediction")
    parser.add_argument("--task", type=str, default="pdbbind", choices=["pdbbind", "chembl"],
                        help="downstream task used in stage 3")

    # ---------------- model architecture ----------------
    parser.add_argument("-multitask", type=bool, default=True, help="always multitask here")
    parser.add_argument("-fusion_type", type=str, default="bidirectional",
                        help="bidirectional, cross_attention, co_attention, multi_head_cross, ban")
    parser.add_argument("-final_predict_layer", type=str, default="OuterProductLinear",
                        help="ConcatMLP, HadamardMLP, DiffAbsConcatMLP, BilinearOuterProduct, "
                             "BilinearProjection, GatedFusion, EnsembleFusion, OuterProductLinear")
    parser.add_argument("-name", type=str, default="run", help="experiment name")

    # ---------------- decoder / fingerprint generator ----------------
    parser.add_argument("-decoder_layers", type=int, default=10)
    parser.add_argument("-num_decoder_head", type=int, default=8)
    parser.add_argument("-ffn_depth", type=int, default=1148)
    parser.add_argument("-model_depth", type=int, default=80)
    parser.add_argument("-decoder_prot_max_len", type=int, default=1024)
    parser.add_argument("-vocab_size", type=int, default=287)

    # ---------------- loss weights ----------------
    parser.add_argument("-affinity_weight", type=float, default=1.0,
                        help="weight of the affinity (MSE) loss")
    parser.add_argument("-sequence_weight", type=float, default=1.0,
                        help="weight of the fingerprint-generation (CE) loss")

    # ---------------- GAT encoder ----------------
    parser.add_argument("-radius", type=int, default=1, help="molecular-graph radius")
    parser.add_argument("-gat_dim", type=int, default=100)
    parser.add_argument("-comp_dim", type=int, default=100)
    parser.add_argument("-prot_dim", type=int, default=100)
    parser.add_argument("-latent_dim", type=int, default=100)
    parser.add_argument("-num_head", type=int, default=11, help="GAT attention heads")
    parser.add_argument("-dropout", type=float, default=0.3)
    parser.add_argument("-alpha", type=float, default=0.1, help="LeakyReLU alpha")
    parser.add_argument("-num_layers", type=int, default=4, help="fusion attention layers")

    # ---------------- data paths (edit to your own) ----------------
    parser.add_argument("-prot_esmc_embed", type=str, default="/path/to/esmc_embeddings.pkl",
                        help="precomputed protein embeddings")
    parser.add_argument("-dataset", type=str, default="pdbbind", help="dataset name")
    parser.add_argument("-train_csv_data_path", type=str, default="/path/to/train.csv")
    parser.add_argument("-test_csv_data_path", type=str, default="/path/to/test.csv")
    parser.add_argument("-train_dataset_path", type=str, default="/path/to/train.h5")
    parser.add_argument("-test_dataset_path", type=str, default="/path/to/test.h5")
    parser.add_argument("-vocab", type=str, default="/path/to/vocab.pkl")
    parser.add_argument("--dict_prefix", type=str, default="",
                        help="prefix for the atom/bond/fingerprint dictionary files")

    # ---------------- optimizer / schedule ----------------
    parser.add_argument("-lr", type=float, default=0.0005)
    parser.add_argument("-step_size", type=int, default=100)
    parser.add_argument("-gamma", type=float, default=0.5)
    parser.add_argument("-batch_size", type=int, default=16)
    parser.add_argument("-num_epochs", type=int, default=200)
    parser.add_argument("-seed", type=int, default=42)

    # ---------------- finetuning ----------------
    parser.add_argument("-finetune_from", type=str, default="",
                        help="path to a pretrained checkpoint; empty = train from scratch")
    parser.add_argument("-finetune_lr", type=float, default=1e-5,
                        help="learning rate used when -finetune_from is set")
    parser.add_argument("-freeze_layers", type=str, default="",
                        help="JSON string of layers/modules to freeze, e.g. "
                             "'{\"transformer_decoder\": true, \"generator\": true}' "
                             "or '{\"decoder_layers\": [0,1,2,3,4,5,6]}'")

    # ---------------- runtime ----------------
    parser.add_argument("-mode", type=str, default="gpu", help="gpu/cpu")
    parser.add_argument("-cuda", type=str, default="0")
    parser.add_argument("-verbose", type=int, default=1)
    parser.add_argument("-save_checkpoint_every", type=int, default=1)
    parser.add_argument("-checkpoint_path", type=str, default="../checkpoints")

    params, _ = parser.parse_known_args()
    return params
