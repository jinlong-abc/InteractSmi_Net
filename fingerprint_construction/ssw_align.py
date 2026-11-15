"""SSW sequence alignment (step 06 -> seq_align.txt).

Aligns the target/query FASTA files produced by step 06 and writes
``data/pdbbind_seq_align.txt`` in the format parsed by ``common/ssw.py``.

The alignment library used here is the ``ssw`` package
(``pip install ssw``, wrapping the Complete-Striped-Smith-Waterman library).
This script is a *reference implementation*: it reproduces the on-disk format
so ``common/ssw.py`` can parse it, but you may use any Smith-Waterman aligner
that emits the same format.

Output format (one record per target/query pair)::

    target_name: <target header>
    query_name: <query header>
    optimal_alignment_score: <s>\\tsuboptimal_alignment_score: <s>\\tstrand: +\\ttarget_begin: <t0>\\ttarget_end: <t1>\\tquery_begin: <q0>\\tquery_end: <q1>
    <blank>
    Target:       <t0>\\t<target seq with gaps>\\t<end>
                  \\t<match line: "|" for match, space otherwise>
    Query:        <q0>\\t<query seq with gaps>\\t<end>
    <blank>
    ... (Target/align/Query groups wrap every ~54 aligned columns)

``common/ssw.py`` only consumes ``target_name``, ``query_name``,
``optimal_alignment_score`` (``target_begin`` / ``query_begin``), and the
``Target`` / align / ``Query`` sequence chunks -- so those must match exactly.
"""

import argparse
import re

try:
    import ssw
except ImportError:  # pragma: no cover
    ssw = None


def parse_fasta(path):
    """Return a list of (header, sequence) from a FASTA file."""
    records = []
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def render_alignment(target, query, alignment):
    """Build gap-aligned target/query strings and the match ('|') line.

    ``alignment.cigar_string`` is processed into M/I/D runs (SSW convention:
    M = aligned, I = gap in target, D = gap in query).
    """
    t_aln, q_aln, match = [], [], []
    ti, qi = 0, 0
    for n, op in re.findall(r"(\d+)([MID])", alignment.cigar_string):
        n = int(n)
        if op == "M":
            for _ in range(n):
                t, q = target[ti], query[qi]
                t_aln.append(t)
                q_aln.append(q)
                match.append("|" if t == q else " ")
                ti += 1
                qi += 1
        elif op == "I":  # gap in target
            t_aln.extend("-" * n)
            q_aln.extend(query[qi:qi + n])
            match.extend(" " * n)
            qi += n
        elif op == "D":  # gap in query
            t_aln.extend(target[ti:ti + n])
            q_aln.extend("-" * n)
            match.extend(" " * n)
            ti += n
    return "".join(t_aln), "".join(q_aln), "".join(match)


def write_record(out, target_name, query_name, t_aln, q_aln, match,
                 target_start, query_start, optimal, suboptimal):
    out.write(f"target_name: {target_name}\n")
    out.write(f"query_name: {query_name}\n")
    out.write(
        f"optimal_alignment_score: {optimal}\t"
        f"suboptimal_alignment_score: {suboptimal}\t"
        f"strand: +\t"
        f"target_begin: {target_start}\t"
        f"target_end: {target_start + len(t_aln.replace('-', '')) - 1}\t"
        f"query_begin: {query_start}\t"
        f"query_end: {query_start + len(q_aln.replace('-', '')) - 1}\n"
    )
    out.write("\n")

    # Wrap the alignment at 54 aligned columns, mirroring the on-disk format.
    width = 54
    t_pos, q_pos = target_start, query_start
    for i in range(0, len(t_aln), width):
        t_chunk = t_aln[i:i + width]
        q_chunk = q_aln[i:i + width]
        m_chunk = match[i:i + width]
        t_end = t_pos + len(t_chunk.replace("-", "")) - 1
        q_end = q_pos + len(q_chunk.replace("-", "")) - 1
        out.write(f"Target:       {t_pos}\t{t_chunk}\t{t_end}\n")
        out.write(f"              \t{m_chunk}\n")
        out.write(f"Query:        {q_pos}\t{q_chunk}\t{q_end}\n")
        out.write("\n")
        t_pos += len(t_chunk.replace("-", ""))
        q_pos += len(q_chunk.replace("-", ""))


def main():
    parser = argparse.ArgumentParser(description="Run SSW alignment for the fingerprint pipeline.")
    parser.add_argument("--target", required=True, help="target FASTA (step 06 output)")
    parser.add_argument("--query", required=True, help="query FASTA (step 06 output)")
    parser.add_argument("--output", default="pdbbind_seq_align.txt")
    parser.add_argument("--match", type=int, default=2)
    parser.add_argument("--mismatch", type=int, default=-3)
    parser.add_argument("--gap_open", type=int, default=5)
    parser.add_argument("--gap_extend", type=int, default=2)
    args = parser.parse_args()

    if ssw is None:
        raise SystemExit("The 'ssw' package is not installed. Run: pip install ssw")

    targets = parse_fasta(args.target)
    queries = parse_fasta(args.query)
    if len(targets) != len(queries):
        raise SystemExit(
            f"target/query FASTA lengths differ: {len(targets)} vs {len(queries)}"
        )

    aligner = ssw.Aligner(
        match=args.match,
        mismatch=args.mismatch,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
    )

    with open(args.output, "w") as out:
        for (t_name, t_seq), (q_name, q_seq) in zip(targets, queries):
            alignment = aligner.align(t_seq, q_seq)
            t_aln, q_aln, match = render_alignment(t_seq, q_seq, alignment)
            write_record(
                out, t_name, q_name, t_aln, q_aln, match,
                alignment.target_begin + 1, alignment.query_begin + 1,
                alignment.optimal_score, alignment.sub_optimal_score,
            )

    print(f"Wrote alignment to {args.output}")


if __name__ == "__main__":
    main()
