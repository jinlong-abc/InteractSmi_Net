# PDBBind interaction-fingerprint pipeline (proLIF + closest-atoms)

A reproducible pipeline that turns a raw [PDBBind](http://www.pdbbind.org.cn/)
dataset into protein–ligand **interaction fingerprints** and **interaction-fusion
sequences** (AtomIdx `IntSeq` / `Fusion SMILES`).

The fingerprint combines two complementary definitions, merged as
**prolif_closest**:

* **proLIF** — rule-based interaction fingerprints (hydrophobic, H-bond, ionic,
  π-stacking, van der Waals contacts, ...), and
* **closest-atoms** — the closest protein residue to each ligand atom within a
  5.0 Å cut-off.

The interaction sequence is then rendered two ways:

* `IntSeq` — a whitespace-separated token list `{atom aa position interaction}`, and
* `Fusion SMILES` — the ligand SMILES with `{aa position interaction}` tags
  inserted after the interacting atoms.

Both are numbered by **absolute** residue position (`IntSeq_1`, `Fusion_SMILES`)
and by **relative** position — the *n*-th occurrence of that amino acid
(`IntSeq_2`, `Fusion_SMILES_2`).

## Pipeline overview

```
PDBBind general-set
        │  01  prepare complexes (mol2/sdf → pdb, merge protein+ligand)
        ▼
   data/complexes/
        ├── 02  extract pdbid→UniProt map   (INDEX_general_PL_name.2020)
        │        03  SMILES + affinity table  (INDEX_general_PL.2020 + ligand pdb)
        │        04  add protein sequence     (UniProt idmapping FASTA)
        ├── 05  parse complexes → pickle
        │        06  build SSW input fasta  ──► external SSW → seq_align.txt
        ├── 07  proLIF fingerprint
        ├── 08  closest-atoms fingerprint   ──► 09  map proLIF to UniProt
        ▼
   10  AtomIdx-marked SMILES
   11  build IntSeq_1   (proLIF + closest merged, {atom pdb_res inter})
   12  residue → UniProt position
   13  add amino-acid type, abbreviate interaction
   14  filter invalid residues + <3 interactions
   15  join protein sequence + affinity   →  final_pdbbind_intseqs.csv
        ▼
   16  merge (optional BindingNet high/low/moderate)
   17  de-duplicate similar ligands (Tanimoto)
   18  drop large ligands (long peptides)
   19  render Fusion SMILES (+ SMILES)
   20  clean IntSeq (drop atom index, sort by residue)
   21  relative numbering (n-th occurrence)
   22  one-letter → three-letter amino acids
   23  drop redundant VDW tags + <3 interactions
   24  sort IntSeq_2            →  bindingnet_pdbbind_final.csv
```

## Directory layout

```
config.py            central configuration (edit PDBBIND_DIR etc.)
common/              shared data classes, constants, SSW-output parser
steps/               24 numbered scripts, run in order
data/                intermediate results (created at run time)
external/            UniProt idmapping + pre-computed SSW alignment (see below)
requirements.txt     Python dependencies
```

> `external/` (gitignored) is an optional staging area for the UniProt idmapping
> files and a pre-computed `pdbbind_seq_align.txt`. Point `config.py` at
> wherever you keep these files.

## Requirements

* Python ≥ 3.9 (tested on 3.12)
* The packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> `openbabel` (Python bindings) is only a fallback for ligand conversion in
> step 01; the pipeline runs without it when the RDKit conversion succeeds.

## External inputs

Three things are *not* shipped with this repository:

1. **PDBBind** (v2020 general-set used here). Download from
   [pdbbind.org.cn](http://www.pdbbind.org.cn/) and point `PDBBIND_DIR` at the
   unpacked set (it must contain `INDEX_general_PL.2020`,
   `INDEX_general_PL_name.2020`, and one subdirectory per complex).

2. **UniProt idmapping files** for the protein sequences:
   * a FASTA of UniProtKB sequences (headers like `>sp|P00760|TRY1_BOVIN ...`), and
   * a tab-separated idmapping table with a header row `From`/`Entry`
     (maps the id in `INDEX_general_PL_name.2020` to the current UniProtKB
     accession).

   Both are available from the UniProt idmapping FTP
   (`idmapping_selected.tab` filtered to `From`+`Entry` with a header added, and
   the UniProtKB FASTA). Set `IDMAPPING_FASTA` and `IDMAPPING_TSV`.

3. **SSW sequence alignment** (step 06). Steps 06 writes two FASTA files
   (`pdbbind_ssw_target.fasta`, `pdbbind_ssw_query.fasta`); you run your SSW
   aligner over them and save the result to `data/pdbbind_seq_align.txt` in the
   format consumed by `common/ssw.py` (one record per target with
   `target_name`, `query_name`, `optimal_alignment_score`, `target_begin:`,
   `query_begin:` fields). Any Smith–Waterman implementation that emits this
   format works.

## Configuration

Edit `config.py` (or set the equivalent environment variables):

| Setting | Environment variable | Meaning |
|---|---|---|
| `PDBBIND_DIR` | `PDBBIND_DIR` | PDBBind dataset root (per-complex subdirectories) |
| `PDBBIND_INDEX_DIR` | `PDBBIND_INDEX_DIR` | directory with `INDEX_general_PL*.2020` (defaults to `../index`) |
| `IDMAPPING_FASTA` | `IDMAPPING_FASTA` | UniProt sequences FASTA |
| `IDMAPPING_TSV` | `IDMAPPING_TSV` | UniProt `From`/`Entry` table |
| `BINDINGNET_HIGH/LOW/MODERATE` | same | optional BindingNet intseq CSVs for step 16 |

## Running

Run the steps in order. Steps 01–15 build the PDBBind fingerprints and the
first fusion sequence; steps 16–24 are the downstream refinement.

```bash
python steps/01_prepare_complexes.py
python steps/02_extract_uniprot_ids.py
python steps/03_build_smiles_affinity.py
python steps/04_add_protein_sequences.py
python steps/05_parse_pdb_complexes.py
python steps/06_build_ssw_input.py
# ... run your SSW aligner -> data/pdbbind_seq_align.txt ...
python steps/07_run_prolif.py
python steps/08_find_closest_atoms.py
python steps/09_map_prolif_to_uniprot.py
python steps/10_atomidx_smiles.py
python steps/11_build_intseq.py
python steps/12_convert_residue_to_uniprot.py
python steps/13_add_aa_type.py
python steps/14_filter_intseq_errors.py
python steps/15_finalize_intseq.py
# phase 2 (BindingNet merge is optional)
python steps/16_merge_datasets.py
python steps/17_filter_similar_ligands.py
python steps/18_filter_molecular_weight.py
python steps/19_to_fusion_smiles.py
python steps/20_remove_out_of_range_atom_idx.py
python steps/21_relative_numbering.py
python steps/22_one2three.py
python steps/23_remove_vdw_conflicts.py
python steps/24_sort_interactions.py
```

Or use the convenience script:

```bash
bash run_all.sh
```

## Output columns

The final table (`data/bindingnet_pdbbind_final.csv`) has these columns:

| Column | Meaning |
|---|---|
| `ID` | complex identifier |
| `UniProt` | protein UniProt accession |
| `Assay Result` | binding affinity (normalised to nM) |
| `Normal_SMILES` | ligand SMILES |
| `Mapped_SMILES` | ligand SMILES with atom-map numbers |
| `SMILES` | de-mapped SMILES |
| `Fusion_SMILES` | SMILES with `{aa position interaction}` tags (absolute, 3-letter) |
| `Fusion_SMILES_2` | same, relative (n-th occurrence) numbering |
| `IntSeq_1` | token list `{atom aa position interaction}` (atom order) |
| `IntSeq_2` | token list sorted by residue (relative numbering) |
| `Sequence` | protein sequence |

Interaction tokens: `Hy` hydrophobic, `HB` hydrogen bond, `Pi` π-stacking,
`PC` π-cation, `IB` ionic, `VDW` van der Waals contact.

## Known limitations

* Multi-chain complexes: the residue→UniProt mapping uses the single chain with
  the most interactions; interactions on other chains are ignored.
* The interaction pickles (steps 07–09) are Python `pickle` files; regenerate
  them in the same environment (they depend on the classes in `common/`).

## License

MIT. See `LICENSE`.
