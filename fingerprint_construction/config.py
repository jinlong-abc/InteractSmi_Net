"""Central configuration for the PDBBind interaction-fingerprint pipeline.

Every file path used by the pipeline derives from this module. Point the
"User-editable settings" at your local files, then run the numbered scripts in
``steps/`` in order (``python steps/01_*.py``, ...). Intermediate results are
written under ``data/`` next to this file; nothing is written outside the
repository unless you change the paths below.

Required input
--------------
A PDBBind dataset (general-set or refined-set). Its root directory must contain

* ``INDEX_general_PL.2020``       -- affinity + resolution index
* ``INDEX_general_PL_name.2020``  -- PDB-id -> UniProt name index
* one subdirectory per complex, e.g. ``1a0q/`` holding ``1a0q_protein.pdb``
  and ``1a0q_ligand.sdf`` (or ``.mol2``).

Protein sequences are looked up from a UniProt idmapping FASTA/TSV pair
(steps 04 and 06). Both files are available from the UniProt idmapping FTP
(e.g. ``idmapping_selected.tab`` and the UniProtKB FASTA); see README.

Optional input
--------------
Steps 16-24 can merge BindingNet interaction sequences alongside PDBBind. Set
``BINDINGNET_*`` to those CSV files (or leave empty to process PDBBind only).
"""

import os

# Repository root (this file lives at the top level of the pipeline).
ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# User-editable settings (also overridable via environment variables)
# ---------------------------------------------------------------------------

# Path to the PDBBind dataset root (contains one subdirectory per complex).
# Set this to your own copy, or export PDBBIND_DIR.
PDBBIND_DIR = os.environ.get("PDBBIND_DIR", "/path/to/pdbbind")

# Directory holding the PDBBind INDEX files (defaults to the ``index/`` folder
# next to the set, as in the standard PDBBind v2020 layout).
PDBBIND_INDEX_DIR = os.environ.get(
    "PDBBIND_INDEX_DIR", os.path.join(os.path.dirname(PDBBIND_DIR), "index")
)

# UniProt idmapping files (see README for how to obtain them).
IDMAPPING_FASTA = os.environ.get("IDMAPPING_FASTA", "/path/to/idmapping.fasta")
IDMAPPING_TSV = os.environ.get("IDMAPPING_TSV", "/path/to/idmapping.tsv")

# Optional BindingNet intseq inputs for step 16 (CSV with the same 7 columns as
# step 15's output). Leave empty to skip.
BINDINGNET_HIGH = os.environ.get("BINDINGNET_HIGH", "")
BINDINGNET_LOW = os.environ.get("BINDINGNET_LOW", "")
BINDINGNET_MODERATE = os.environ.get("BINDINGNET_MODERATE", "")

# ---------------------------------------------------------------------------
# Derived paths -- do not edit
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # ensure the runtime dir always exists

# Step 01 output: prepared protein/ligand/complex PDBs, one subdir per complex.
COMPLEX_DIR = os.path.join(DATA_DIR, "complexes")

# Index files, expected inside PDBBIND_INDEX_DIR.
INDEX_PL = os.path.join(PDBBIND_INDEX_DIR, "INDEX_general_PL.2020")
INDEX_PL_NAME = os.path.join(PDBBIND_INDEX_DIR, "INDEX_general_PL_name.2020")


def _data(name):
    return os.path.join(DATA_DIR, name)


# Step 02
PDBID_UNIPROT_MAP = _data("pdbid_uniprotid_map.csv")   # pdbid, uniprot-name
UNIPROT_LIST = _data("uniprotid.csv")

# Step 03 / 04
SMILES_AFFINITY = _data("pdbbind_smiles_affinity.csv")  # pdbid, UniProt, SMILES, affinity
WITH_SEQ = _data("pdbbind_with_seq.csv")                # + Sequence

# Step 05
ALL_PDBS_PKL = _data("pdbbind_all_pdbs.pkl")

# Step 06 (SSW sequence alignment; input fasta generated, alignment produced by
# the external ``ssw`` binary into SEQ_ALIGN).
SSW_TARGET_FASTA = _data("pdbbind_ssw_target.fasta")
SSW_QUERY_FASTA = _data("pdbbind_ssw_query.fasta")
SEQ_ALIGN = _data("pdbbind_seq_align.txt")

# Step 07 / 08 / 09
PROLIF_PKL = _data("prolif_interactions.pkl")
CLOSEST_PKL = _data("closest_interactions.pkl")
PROLIF_MAPPED_PKL = _data("prolif_interactions_mapped.pkl")

# Step 10 / 11 / 12 / 13
MAPPED_SMILES = _data("pdbbind_mapped_smiles.csv")                 # PDBID, Normal_SMILES, Mapped_SMILES
INTSEQ_CSV = _data("pdbbind_mapped_smiles_with_intseq.csv")        # + IntSeq_1
INTSEQ_UNIPROT_CSV = _data("pdbbind_mapped_smiles_with_intseq_2.csv")
INTSEQ_AA_CSV = _data("pdbbind_mapped_smiles_with_intseq_3.csv")

# Step 14 / 15
INTSEQ_FILTERED = _data("pdbbind_intseqs_filtered.csv")
FINAL_INTSEQ = _data("final_pdbbind_intseqs.csv")                  # 7 columns

# Step 16
MERGED_INTSEQ = _data("bindingnet_pdbbind_intseqs_all.csv")

# Step 17 / 18
SIM_FILTERED = _data("bindingnet_pdbbind_intseqs_all_filtered_sim.csv")
MW_FILTERED = _data("bindingnet_pdbbind_intseqs_all_filtered_mw.csv")

# Step 19
FUSION_SMILES_CSV = _data("bindingnet_pdbbind_intseqs_fusion_smiles.csv")

# Step 20
ATOM_IDX_FILTERED = _data("bindingnet_pdbbind_intseqs_atomidx_filtered.csv")

# Step 21
RELATIVE_NUMBERED = _data("bindingnet_pdbbind_intseqs_relative.csv")

# Step 22
ONE2THREE = _data("bindingnet_pdbbind_intseqs_one2three.csv")

# Step 23 / 24
VDW_FILTERED = _data("bindingnet_pdbbind_intseqs_vdw_filtered.csv")
FINAL_OUTPUT = _data("bindingnet_pdbbind_final.csv")
