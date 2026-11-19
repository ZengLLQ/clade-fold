"""Tests for Gotoh pairwise/progressive MSA, guide trees, and conservation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.msa_engine import (
    AlignmentConfig,
    align_pairwise_gotoh,
    build_progressive_msa,
    compute_conservation_profile,
    compute_henikoff_weights,
    kimura_distance_from_aligned,
    neighbor_joining,
    pairwise_identity_matrix,
    summarise_msa,
    trim_gappy_columns,
    upgma_tree,
)
from clade_fold.phylogenetics import robinson_foulds_distance
from clade_fold.types import MultipleSequenceAlignment, TreeNode


def test_gotoh_pairwise_alignment_recovers_indel() -> None:
    s1 = "MKTAYIAKQRQISFVKSHFSRQ"
    s2 = "MKTAYIAKQISFVKSHFSRQ"  # 2-residue deletion ("RQ")
    a1, a2, score, ident = align_pairwise_gotoh(s1, s2, matrix_name="BLOSUM62")
    assert len(a1) == len(a2)
    assert a1.replace("-", "") == s1
    assert a2.replace("-", "") == s2
    assert a2.count("-") == 2
    assert ident > 0.95
    assert score > 50.0


def test_neighbor_joining_recovers_exact_additive_4taxon_tree() -> None:
    # Additive tree: ((A:0.1, B:0.2):0.3, (C:0.15, D:0.25):0.0)
    labels = ["A", "B", "C", "D"]
    D = np.array([
        [0.0, 0.30, 0.55, 0.65],
        [0.30, 0.0, 0.65, 0.75],
        [0.55, 0.65, 0.0, 0.40],
        [0.65, 0.75, 0.40, 0.0],
    ], dtype=np.float64)
    nj = neighbor_joining(D, labels)
    ref = TreeNode(
        children=[
            TreeNode(children=[TreeNode("A", 0.1), TreeNode("B", 0.2)], branch_length=0.15),
            TreeNode(children=[TreeNode("C", 0.15), TreeNode("D", 0.25)], branch_length=0.15),
        ]
    )
    rf, _ = robinson_foulds_distance(nj, ref)
    assert rf == 0
    assert nj.total_branch_length() == pytest.approx(1.0, abs=1e-5)

    up = upgma_tree(D, labels)
    assert len(up.leaves()) == 4


def test_henikoff_weights_downweight_duplicates() -> None:
    msa = MultipleSequenceAlignment(
        names=["dup1", "dup2", "dup3", "outgroup"],
        sequences=["ACDEFG", "ACDEFG", "ACDEFG", "WXYZIK".replace("X", "H").replace("Z", "M")],
    )
    w = compute_henikoff_weights(msa.matrix)
    assert float(np.sum(w)) == pytest.approx(4.0, abs=1e-6)
    # Outgroup carries strictly greater weight than any individual duplicate
    assert w[3] > w[0]


def test_progressive_msa_with_refinement_and_trimming() -> None:
    records = [
        ("s1", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLS"),
        ("s2", "MVLSPADKTNVKAAWGKVAHAGEYGAEALERMFLS"),
        ("s3", "MVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFAS"),
        ("s4", "MVLSAADKNNVKGIFTKIAGHAEEYGAETLERMFTT"),
    ]
    msa = build_progressive_msa(
        records, AlignmentConfig(guide_method="nj", refinement_rounds=2)
    )
    assert msa.n_seqs == 4
    assert all(seq.replace("-", "") == orig for seq, (_, orig) in zip(msa.sequences, records))

    cons = compute_conservation_profile(msa)
    assert cons.shape == (msa.length,)
    assert np.all((cons >= 0.0) & (cons <= 1.0))

    ident = pairwise_identity_matrix(msa)
    assert np.allclose(np.diag(ident), 1.0)
    _, kd = kimura_distance_from_aligned(msa.sequences[0], msa.sequences[2])
    assert kd > 0.0

    trimmed = trim_gappy_columns(msa, max_gap_fraction=0.3)
    assert trimmed.length <= msa.length
    stats = summarise_msa(msa)
    assert stats["n_seqs"] == 4.0
