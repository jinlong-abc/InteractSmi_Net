"""Parser for the SSW sequence-alignment output.

The external ``ssw`` tool writes ``pdbbind_seq_align.txt``; these helpers turn
that into a ``pdbid -> {chain: [uniprot index per PDB residue]}`` mapping that
steps 08/09 use to translate PDB residue numbers to UniProt positions.
"""


def get_result_dict(align_path):
    """Parse the SSW output into {target_name: (target, query, align, t0, q0)}."""
    result = {}
    seq_target, seq_query, align = "", "", ""
    target_name = query_name = ""
    target_start = query_start = 0

    with open(align_path) as f:
        lines = f.readlines()

    i = -1
    for line in lines:
        i += 1
        if i % 4 == 0:
            if "target_name" in line:
                if len(seq_target) != 0:
                    result[target_name] = (
                        seq_target, seq_query, align, target_start, query_start,
                    )
                target_name = line.strip().split(" ")[-1]
                seq_target, seq_query, align = "", "", ""
            else:
                seq_target += line.split("\t")[1]
        elif i % 4 == 1:
            if "query_name" in line:
                query_name = line.strip().split(" ")[-1]
            else:
                align += line.strip("\n").split("\t")[1]
        elif i % 4 == 2:
            if "optimal_alignment_score" in line:
                for item in line.strip().split("\t"):
                    if item.split(" ")[0] == "target_begin:":
                        target_start = int(item.split(" ")[1])
                    elif item.split(" ")[0] == "query_begin:":
                        query_start = int(item.split(" ")[1])
            else:
                seq_query += line.split("\t")[1]

    # flush the final record
    if len(seq_target) != 0 and target_name:
        result[target_name] = (
            seq_target, seq_query, align, target_start, query_start,
        )
    return result


def _seq_with_gap_to_idx(seq):
    idx_list = []
    i = 0
    for aa in seq:
        if aa == "-":
            idx_list.append(-1)
        else:
            idx_list.append(i)
            i += 1
    return idx_list


def _get_target_idx(target_idx_list, query_idx_list, align, target_start, query_start):
    pdb_to_uniprot_idx = [-1] * (target_start - 1)
    for i in range(len(target_idx_list)):
        if target_idx_list[i] != -1:
            if align[i] == "|" and query_idx_list[i] != -1:
                pdb_to_uniprot_idx.append(query_idx_list[i] + query_start - 1)
            else:
                pdb_to_uniprot_idx.append(-1)
    return pdb_to_uniprot_idx


def get_pdb_to_uniprot_map(result_dict, min_identity=0.9):
    """Return {pdbid: {chain: [uniprot_idx, ...]}} for well-aligned chains."""
    pdb_to_uniprot_map = {}
    for name in result_dict:
        pdbid = name.split("_")[0]
        chain = name.split("_")[-1]
        seq_target, seq_query, align, target_start, query_start = result_dict[name]

        target_nogap = seq_target.replace("-", "")
        ratio = float(align.count("|")) / float(len(target_nogap)) if target_nogap else 0.0
        if ratio < min_identity:
            continue

        target_idx_list = _seq_with_gap_to_idx(seq_target)
        query_idx_list = _seq_with_gap_to_idx(seq_query)
        pdb_to_uniprot_idx = _get_target_idx(
            target_idx_list, query_idx_list, align, target_start, query_start
        )
        pdb_to_uniprot_map.setdefault(pdbid, {})[chain] = pdb_to_uniprot_idx

    return pdb_to_uniprot_map
