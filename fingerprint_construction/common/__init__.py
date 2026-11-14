"""Shared data structures and constants for the pipeline."""

from .data_classes import (
    Residue,
    Chain,
    Atom,
    Ligand,
    ProteinComplex,
    ProteinLigandInteractionClosest,
    ProteinLigandInteractionProlif,
)
from .constants import AA1_TO_AA3, AA3_TO_AA1, INTERACTION_ABBREV

__all__ = [
    "Residue",
    "Chain",
    "Atom",
    "Ligand",
    "ProteinComplex",
    "ProteinLigandInteractionClosest",
    "ProteinLigandInteractionProlif",
    "AA1_TO_AA3",
    "AA3_TO_AA1",
    "INTERACTION_ABBREV",
]
