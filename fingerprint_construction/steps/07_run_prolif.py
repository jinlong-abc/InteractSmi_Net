"""Step 07: compute proLIF interaction fingerprints for every complex.

Reads each prepared protein/ligand PDB pair, runs proLIF
(``prolif.Fingerprint``, 6.0 A vicinity cutoff), and stores one
``ProteinLigandInteractionProlif`` record per complex. Output pickle:
``prolif_interactions.pkl``.
"""

import os
import sys
import pickle
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import prolif
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import SaltRemover
from tqdm import tqdm

from common.data_classes import ProteinLigandInteractionProlif

RDLogger.DisableLog("rdApp.*")

_WATER_RESIDUES = {"HOH", "WAT", "H2O", "OH2", "NA", "CL", "MG", "CA", "K", "ZN"}


def safe_read_pdb(pdb_file, protein=True):
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"Failed to parse PDB file: {pdb_file}")

    if protein:
        mol = SaltRemover.SaltRemover().StripMol(mol)
        remove_atoms = []
        for atom in mol.GetAtoms():
            res_info = atom.GetPDBResidueInfo()
            if res_info and res_info.GetResidueName().strip() in _WATER_RESIDUES:
                remove_atoms.append(atom.GetIdx())
        rw_mol = Chem.RWMol(mol)
        for idx in sorted(set(remove_atoms), reverse=True):
            rw_mol.RemoveAtom(idx)
        mol = rw_mol.GetMol()

    return mol


def analyze_interactions(protein_pdb, ligand_pdb, pdbid):
    protein_mol = safe_read_pdb(protein_pdb, protein=True)
    ligand_mol = safe_read_pdb(ligand_pdb, protein=False)

    protein = prolif.Molecule(protein_mol)
    ligand = prolif.Molecule(ligand_mol)

    fp = prolif.Fingerprint(vicinity_cutoff=6.0)
    fp.run_from_iterable([ligand], protein, progress=False)

    records = []
    for frame_idx, interactions in fp.ifp.items():
        for (_, prot_res), interaction_data in interactions.items():
            prot_number = prot_res.number
            prot_chain = prot_res.chain
            for itype, details in interaction_data.items():
                for detail in details:
                    records.extend(
                        (prot_number, prot_chain, lig_atom, itype)
                        for lig_atom in detail["indices"]["ligand"]
                    )

    if not records:
        return None

    residues, chains, indices, types = zip(*records)
    return ProteinLigandInteractionProlif(
        chemblid_all=pdbid,
        protein_residues=list(residues),
        residues_chain=list(chains),
        ligand_indices=list(indices),
        interaction_types=list(types),
        uniprot_idx_list=[],  # filled in by step 09 (map_prolif_to_uniprot)
    )


def process_one(pdbid):
    prot_file = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_protein.pdb")
    lig_file = os.path.join(config.COMPLEX_DIR, pdbid, f"{pdbid}_ligand.pdb")
    try:
        return analyze_interactions(prot_file, lig_file, pdbid), None
    except Exception:
        return None, pdbid


def main():
    pdbids = [d for d in os.listdir(config.COMPLEX_DIR)
              if os.path.isdir(os.path.join(config.COMPLEX_DIR, d))]

    results, errors = [], []
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for result, error in tqdm(pool.imap(process_one, pdbids), total=len(pdbids),
                                  desc="proLIF"):
            if error:
                errors.append(error)
            if result:
                results.append(result)

    with open(config.PROLIF_PKL, "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(config.DATA_DIR, "prolif_errors.log"), "w") as f:
        f.write("\n".join(errors))

    print(f"proLIF complete: {len(results)} success, {len(errors)} failed.")


if __name__ == "__main__":
    main()
