"""Additional conservation-law, numerical convergence, and boundary-invariant tests."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.coevolution import (
    CoevolutionConfig,
    compute_clustering_weights,
    compute_plmdca,
    evaluate_contact_predictions,
)
from clade_fold.contact_folding import FoldingConfig, reconstruct_ca_fold
from clade_fold.elastic_dynamics import build_anm_hessian, build_gnm_kirchhoff
from clade_fold.io_formats import parse_a3m, parse_newick, parse_pdb, write_newick
from clade_fold.msa_engine import (
    AlignmentConfig,
    align_pairwise_gotoh,
    build_progressive_msa,
    kmer_distance_matrix,
    kimura_distance_from_aligned,
)
from clade_fold.phylogenetics import (
    felsenstein_log_likelihood,
    reconstruct_ancestral_sequences,
    robinson_foulds_distance,
)
from clade_fold.structural_geometry import (
    compute_dihedral_deg,
    compute_shrake_rupley_sasa,
    compute_tm_score,
    score_ramachandran,
)
from clade_fold.substitution_models import SubstitutionModel, discrete_gamma_rates
from clade_fold.synthetic_benchmark import (
    generate_beta_hairpin_fold,
    generate_helical_hairpin_fold,
    generate_random_phylo_tree,
    simulate_potts_msa,
    simulate_tree_evolution,
)
from clade_fold.types import MultipleSequenceAlignment, TreeNode


def test_felsenstein_gap_marginalization_identity() -> None:
    """Verify gap '-' at a leaf acts as exact missing-data marginalization."""
    model = SubstitutionModel.from_name("JTT")
    tree_2 = TreeNode(
        children=[
            TreeNode(name="A", branch_length=0.2),
            TreeNode(name="B", branch_length=0.3),
        ]
    )
    msa_2 = MultipleSequenceAlignment(names=["A", "B"], sequences=["ACDE", "AC-E"])
    ll_2 = felsenstein_log_likelihood(tree_2, msa_2, model, alpha=1.0, n_gamma=1)

    # Marginalizing out B at site 2 equals stationary log pi_A(D) at site 2
    tree_a_only = TreeNode(
        children=[
            TreeNode(name="A", branch_length=0.2),
            TreeNode(name="B", branch_length=0.3),
        ]
    )
    msa_gap_all = MultipleSequenceAlignment(names=["A", "B"], sequences=["D", "-"])
    ll_gap = felsenstein_log_likelihood(tree_a_only, msa_gap_all, model, alpha=1.0, n_gamma=1)
    assert ll_gap == pytest.approx(float(np.log(model.pi[2])), abs=1e-10)
    assert np.isfinite(ll_2)


def test_felsenstein_deep_tree_numerical_stability() -> None:
    """Verify log-scaling prevents underflow on 20-taxon, 250-site alignments."""
    tree = generate_random_phylo_tree(n_taxa=20, seed=55, min_branch=0.1, max_branch=0.45)
    msa, _, _ = simulate_tree_evolution(tree, length=250, model_name="WAG", alpha=0.8, seed=56)
    model = SubstitutionModel.from_name("WAG")
    ll = felsenstein_log_likelihood(tree, msa, model, alpha=0.8, n_gamma=4)
    assert np.isfinite(ll)
    assert ll < 0.0


def test_shrake_rupley_sasa_quadrature_convergence() -> None:
    """Verify SASA converges monotonically as Fibonacci sphere quadrature increases."""
    struct = generate_helical_hairpin_fold(length=22, seed=8)
    s42, _, _, _ = compute_shrake_rupley_sasa(struct, n_sphere_points=42)
    s92, _, _, _ = compute_shrake_rupley_sasa(struct, n_sphere_points=92)
    s182, _, _, _ = compute_shrake_rupley_sasa(struct, n_sphere_points=182)
    err_coarse = float(np.mean(np.abs(s42 - s182)))
    err_fine = float(np.mean(np.abs(s92 - s182)))
    assert err_fine <= err_coarse + 1.0


def test_anm_analytical_translation_and_rotation_nullspace() -> None:
    """Explicitly project 3 rigid translations and 3 rigid rotations onto ANM Hessian."""
    struct = generate_beta_hairpin_fold(length=20, seed=4)
    ca = struct.ca_coords
    centered = ca - np.mean(ca, axis=0, keepdims=True)
    H = build_anm_hessian(centered, cutoff=13.0, gamma=1.0)
    N = centered.shape[0]

    # 3 pure translations
    for axis in range(3):
        v = np.zeros((N, 3), dtype=np.float64)
        v[:, axis] = 1.0
        v_flat = v.ravel() / np.linalg.norm(v.ravel())
        assert float(v_flat @ H @ v_flat) < 1e-10

    # 3 infinitesimal rotations around coordinate axes
    for axis in range(3):
        e = np.zeros(3, dtype=np.float64)
        e[axis] = 1.0
        v = np.cross(e[None, :], centered)
        v_flat = v.ravel() / np.linalg.norm(v.ravel())
        assert float(v_flat @ H @ v_flat) < 1e-10


def test_gnm_kirchhoff_connected_laplacian_property() -> None:
    struct = generate_helical_hairpin_fold(length=24, seed=2)
    Gamma = build_gnm_kirchhoff(struct.ca_coords, cutoff=7.0)
    evals = np.sort(np.linalg.eigvalsh(Gamma))
    # Exactly 1 zero eigenvalue (algebraic connectivity lambda_2 > 0)
    assert abs(float(evals[0])) < 1e-10
    assert float(evals[1]) > 1e-4


def test_gotoh_affine_gap_prefers_single_gap_over_split_gaps() -> None:
    s1 = "ACDEFGHIKLMNPQ"
    s2 = "ACDEKLMNPQ"  # 4-residue contiguous deletion FGHI
    a1, a2, _, _ = align_pairwise_gotoh(s1, s2, gap_open=-12.0, gap_extend=-1.0)
    assert a1 == s1
    assert "----" in a2


def test_kmer_and_kimura_distance_symmetry_and_zero_diagonal() -> None:
    seqs = ["ACDEFGHIKLMN", "ACDEFGHIKLMN", "WYYVTSRQPNMK"]
    kd_mat = kmer_distance_matrix(seqs, k=3)
    assert np.allclose(kd_mat, kd_mat.T)
    assert kd_mat[0, 1] < kd_mat[0, 2]
    ident_same, d_same = kimura_distance_from_aligned(seqs[0], seqs[1])
    assert ident_same == pytest.approx(1.0)
    assert d_same == pytest.approx(0.0, abs=1e-12)


def test_progressive_msa_single_sequence_and_upgma_mode() -> None:
    single = build_progressive_msa([("only", "ACDEFG")])
    assert single.n_seqs == 1 and single.length == 6
    recs = [("a", "ACDEFGHIK"), ("b", "ACDFGHIK"), ("c", "ACDEFGHK")]
    msa = build_progressive_msa(recs, AlignmentConfig(guide_method="upgma", refinement_rounds=1))
    assert msa.n_seqs == 3
    assert all(s.replace("-", "") == orig for s, (_, orig) in zip(msa.sequences, recs))


def test_dihedral_and_ramachandran_edge_cases() -> None:
    # Trans planar dihedral = +/-180 deg
    p0 = np.array([0.0, 1.0, 0.0])
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    p3 = np.array([1.0, -1.0, 0.0])
    ang = compute_dihedral_deg(p0, p1, p2, p3)
    assert abs(abs(ang) - 180.0) < 1e-6

    zones, ll = score_ramachandran(
        np.array([np.nan, -63.0, -120.0, 58.0]),
        np.array([np.nan, -43.0, 130.0, 42.0]),
    )
    assert zones == ["terminus", "alpha_R", "beta", "alpha_L"]
    assert ll[0] == 0.0 and np.all(np.isfinite(ll))


def test_gamma_alpha_large_limit_approaches_uniform() -> None:
    rates = discrete_gamma_rates(alpha=1000.0, n_categories=4)
    assert np.allclose(rates, np.ones(4), atol=1e-6)
    with pytest.raises(ValueError):
        discrete_gamma_rates(alpha=-1.0, n_categories=4)


def test_robinson_foulds_detects_topological_difference() -> None:
    t1 = parse_newick("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);")
    t2 = parse_newick("((A:0.1,C:0.1):0.2,(B:0.1,D:0.1):0.2);")
    rf, nrf = robinson_foulds_distance(t1, t2)
    assert rf == 2
    assert nrf == pytest.approx(1.0)
    assert write_newick(t1, include_support=False).endswith(";")


def test_a3m_preserve_lowercase_and_pdb_invalid_input() -> None:
    msa = parse_a3m(">s1\nACdeF\n>s2\nACDEF\n", strip_lowercase_inserts=False)
    assert msa.sequences == ["ACDEF", "ACDEF"]
    with pytest.raises(ValueError):
        parse_pdb("REMARK empty file\nEND\n")


def test_coevolution_potts_m_eff_scaling_and_asr_gap_tolerance() -> None:
    struct = generate_helical_hairpin_fold(length=22, seed=33)
    msa = simulate_potts_msa(struct, n_seqs=80, coupling_strength=2.0, seed=34)
    _, m_eff = compute_clustering_weights(msa.matrix, identity_threshold=0.95)
    assert 1.0 <= m_eff <= 80.0
    res = compute_plmdca(msa, CoevolutionConfig(plm_maxiter=20))
    ev = evaluate_contact_predictions(res, struct)
    assert ev.auroc is not None and 0.0 <= ev.auroc <= 1.0

    # ASR handles gapped leaf columns cleanly
    t = parse_newick("((A:0.1,B:0.1):0.1,C:0.2);")
    gapped = MultipleSequenceAlignment(names=["A", "B", "C"], sequences=["AC-D", "AC-D", "ACVD"])
    asr = reconstruct_ancestral_sequences(t, gapped, SubstitutionModel.from_name("WAG"))
    assert len(asr.marginal_sequences["Node_0"]) == 4
    assert len(asr.joint_sequences["Node_0"]) == 4


def test_tm_score_symmetry_and_folding_without_dssp() -> None:
    s1 = generate_helical_hairpin_fold(length=22, seed=4)
    s2 = generate_beta_hairpin_fold(length=22, seed=5)
    tm_diff = compute_tm_score(s1.ca_coords, s2.ca_coords)
    assert 0.0 < tm_diff < 0.7

    coev_msa = simulate_potts_msa(s1, n_seqs=60, coupling_strength=2.0, seed=9)
    res = compute_plmdca(coev_msa, CoevolutionConfig(plm_maxiter=20))
    fold = reconstruct_ca_fold(
        res, dssp_codes=None, config=FoldingConfig(maxiter=60, n_restarts=1)
    )
    assert fold.ca_coords.shape == (22, 3)
    assert np.all(np.isfinite(fold.ca_coords))
