#!/usr/bin/env bash
# Run the PDBBind interaction-fingerprint pipeline end to end.
# Steps run in order; edit config.py first. The SSW alignment step (between 06
# and 07) is manual -- see README.
set -euo pipefail

cd "$(dirname "$0")"

run() {
  echo "==== $1 ===="
  python "$1"
}

run steps/01_prepare_complexes.py
run steps/02_extract_uniprot_ids.py
run steps/03_build_smiles_affinity.py
run steps/04_add_protein_sequences.py
run steps/05_parse_pdb_complexes.py
run steps/06_build_ssw_input.py

echo "==> SSW sequence alignment (step 06 -> 07)"
if [ ! -f data/pdbbind_seq_align.txt ]; then
  if [ -f external/pdbbind_seq_align.txt ]; then
    cp external/pdbbind_seq_align.txt data/pdbbind_seq_align.txt
    echo "    copied pre-computed external/pdbbind_seq_align.txt"
  else
    echo "    data/pdbbind_seq_align.txt not found -- run your SSW aligner, then re-run."
    exit 0
  fi
fi

run steps/07_run_prolif.py
run steps/08_find_closest_atoms.py
run steps/09_map_prolif_to_uniprot.py
run steps/10_atomidx_smiles.py
run steps/11_build_intseq.py
run steps/12_convert_residue_to_uniprot.py
run steps/13_add_aa_type.py
run steps/14_filter_intseq_errors.py
run steps/15_finalize_intseq.py
run steps/16_merge_datasets.py
run steps/17_filter_similar_ligands.py
run steps/18_filter_molecular_weight.py
run steps/19_to_fusion_smiles.py
run steps/20_remove_out_of_range_atom_idx.py
run steps/21_relative_numbering.py
run steps/22_one2three.py
run steps/23_remove_vdw_conflicts.py
run steps/24_sort_interactions.py

echo "Done. Final output: data/bindingnet_pdbbind_final.csv"
