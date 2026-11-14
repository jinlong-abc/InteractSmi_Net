"""Step 01: prepare protein-ligand complex PDBs from the PDBBind set.

For every complex in the PDBBind dataset this script

* converts the ligand (SDF/MOL2) to PDB,
* copies the protein PDB,
* merges protein + ligand into a single ``*_complex.pdb`` (ligand placed on
  chain ``w`` so its chain id never collides with the protein chains).

Output is written to ``config.COMPLEX_DIR``, one subdirectory per complex.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB

try:
    from openbabel import pybel
    _HAS_OPENBABEL = True
except Exception:  # openbabel is optional; used only as a fallback
    _HAS_OPENBABEL = False


def convert_mol2_to_pdb(mol2_file, pdb_file):
    try:
        mol = Chem.MolFromMol2File(mol2_file, removeHs=False, sanitize=False)
        if mol is None:
            return None
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)
        with open(pdb_file, "w") as f:
            f.write(Chem.MolToPDBBlock(mol))
        return pdb_file
    except Exception:
        return None


def convert_sdf_to_pdb(sdf_file, pdb_file):
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    if suppl is None:
        return None
    mol = suppl[0]
    if mol is None:
        return None
    Chem.MolToPDBFile(mol, pdb_file)
    return pdb_file


def convert_sdf_to_pdb_obabel(sdf_file, pdb_file):
    if not _HAS_OPENBABEL:
        return None
    try:
        mols = list(pybel.readfile("sdf", sdf_file))
        if mols:
            mols[0].write("pdb", pdb_file, overwrite=True)
            return pdb_file
    except Exception:
        return None
    return None


def merge_pdb(protein_file, ligand_file, output_file):
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    structure_protein = parser.get_structure("protein", protein_file)
    structure_ligand = parser.get_structure("ligand", ligand_file)

    model_protein = structure_protein[0]
    for model in structure_ligand:
        for chain in model:
            chain.id = "w"  # ligand chain id, distinct from protein chains
            model_protein.add(chain)

    io.set_structure(structure_protein)
    io.save(output_file)


def main():
    os.makedirs(config.COMPLEX_DIR, exist_ok=True)
    success_count = 0
    fail_count = 0

    pdbids = [d for d in os.listdir(config.PDBBIND_DIR)
              if os.path.isdir(os.path.join(config.PDBBIND_DIR, d))]

    for pdbid in pdbids:
        input_dir = os.path.join(config.PDBBIND_DIR, pdbid)
        output_dir = os.path.join(config.COMPLEX_DIR, pdbid)
        os.makedirs(output_dir, exist_ok=True)

        ligand_sdf = os.path.join(input_dir, f"{pdbid}_ligand.sdf")
        ligand_mol2 = os.path.join(input_dir, f"{pdbid}_ligand.mol2")
        protein_pdb = os.path.join(input_dir, f"{pdbid}_protein.pdb")

        ligand_pdb = os.path.join(output_dir, f"{pdbid}_ligand.pdb")
        protein_out = os.path.join(output_dir, f"{pdbid}_protein.pdb")
        complex_pdb = os.path.join(output_dir, f"{pdbid}_complex.pdb")

        if not os.path.exists(protein_pdb):
            print(f"[Skip] protein PDB not found: {protein_pdb}")
            fail_count += 1
            continue
        if not (os.path.exists(ligand_sdf) or os.path.exists(ligand_mol2)):
            print(f"[Skip] no ligand file for: {pdbid}")
            fail_count += 1
            continue

        converted = False
        if os.path.exists(ligand_sdf):
            converted = bool(convert_sdf_to_pdb(ligand_sdf, ligand_pdb)
                             or convert_sdf_to_pdb_obabel(ligand_sdf, ligand_pdb))
        if not converted and os.path.exists(ligand_mol2):
            converted = bool(convert_mol2_to_pdb(ligand_mol2, ligand_pdb))
        if not converted:
            print(f"[Fail] ligand conversion failed for: {pdbid}")
            fail_count += 1
            continue

        # Keep the original ligand files alongside the prepared PDBs.
        if os.path.exists(ligand_sdf):
            os.system(f'cp "{ligand_sdf}" "{output_dir}"')
        if os.path.exists(ligand_mol2):
            os.system(f'cp "{ligand_mol2}" "{output_dir}"')
        os.system(f'cp "{protein_pdb}" "{protein_out}"')

        try:
            merge_pdb(protein_out, ligand_pdb, complex_pdb)
            success_count += 1
        except Exception as e:
            print(f"[Error] merge failed for {pdbid}: {e}")
            fail_count += 1

    total = success_count + fail_count
    print(f"\nSummary: total={total} success={success_count} failed={fail_count}")


if __name__ == "__main__":
    main()
