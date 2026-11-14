"""Amino-acid and interaction-type dictionaries shared by several steps."""

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL", "B": "ASX", "Z": "GLX", "X": "UNK",
    "J": "XLE",
}

# proLIF interaction type name -> compact token used in IntSeq / Fusion SMILES.
INTERACTION_ABBREV = {
    "Hydrophobic": "Hy",
    "HBDonor": "HB",
    "HBAcceptor": "HB",
    "PiStacking": "Pi",
    "Anionic": "IB",
    "Cationic": "IB",
    "CationPi": "PC",
    "PiCation": "PC",
    "VdWContact": "VDW",
    "VDW": "VDW",
}
