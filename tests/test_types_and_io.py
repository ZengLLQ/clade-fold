"""Tests for types, biophysical encodings, and FASTA/A3M/PDB/Newick I/O."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.io_formats import (
    compute_msa_digest,
    compute_virtual_cb,
    parse_a3m,
    parse_fasta,
    parse_newick,
    parse_pdb,
    write_fasta,
    write_newick,
    write_pdb,
)
from clade_fold.synthetic_benchmark import generate_helical_hairpin_fold
from clade_fold.types import GAP_INDEX, decode_sequence, encode_sequence


def test_sequence_encode_decode_roundtrip() -> None:
    seq = "ACDEFGHIKLMNPQRSTVWY-"
    idx = encode_sequence(seq)
    assert idx.shape == (21,)
    assert int(idx[-1]) == GAP_INDEX
    assert decode_sequence(idx) == seq
    # Ambiguous residues map to GAP_INDEX cleanly
    ambig = encode_sequence("BXZ*")
    assert np.all(ambig == GAP_INDEX)


def test_virtual_cb_bond_length_and_tetrahedral_frame() -> None:
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([1.458, 0.0, 0.0])
    c = np.array([2.00, 1.42, 0.0])
    cb = compute_virtual_cb(n, ca, c)
    assert abs(float(np.linalg.norm(cb - ca)) - 1.522) < 1e-6


def test_fasta_and_a3m_parsing() -> None:
    a3m_text = ">seq1\nACdeFG-\n>seq2\nAC-FG\n"
    msa = parse_a3m(a3m_text, strip_lowercase_inserts=True)
    assert msa.n_seqs == 2
    assert msa.length == 5
    assert msa.sequences == ["ACFG-", "AC-FG"]

    fasta_text = write_fasta(list(zip(msa.names, msa.sequences)), line_width=3)
    parsed = parse_fasta(fasta_text)
    assert parsed == [("seq1", "ACFG-"), ("seq2", "AC-FG")]
    assert compute_msa_digest(msa) == compute_msa_digest(msa)


def test_pdb_write_and_parse_roundtrip() -> None:
    struct = generate_helical_hairpin_fold(length=22, seed=5)
    pdb_text = write_pdb(struct)
    reparsed = parse_pdb(pdb_text, name="reloaded")
    assert reparsed.length == struct.length
    assert reparsed.sequence == struct.sequence
    assert np.allclose(reparsed.ca_coords, struct.ca_coords, atol=1e-3)
    assert np.allclose(reparsed.b_factors, struct.b_factors, atol=1e-2)


def test_newick_parse_write_roundtrip_and_errors() -> None:
    nwk = "((Taxon_A:0.120000,Taxon_B:0.080000)0.950:0.050000,'Taxon C':0.200000);"
    tree = parse_newick(nwk)
    leaves = [lf.name for lf in tree.leaves()]
    assert leaves == ["Taxon_A", "Taxon_B", "Taxon C"]
    assert abs(tree.total_branch_length() - 0.45) < 1e-6
    assert tree.children[0].support == pytest.approx(0.95, abs=1e-3)

    roundtrip = write_newick(tree, include_support=True)
    tree2 = parse_newick(roundtrip)
    assert [lf.name for lf in tree2.leaves()] == leaves
    assert abs(tree2.total_branch_length() - 0.45) < 1e-5

    with pytest.raises(ValueError):
        parse_newick("")
    with pytest.raises(ValueError):
        parse_newick("(A:0.1,B:bad);")
