"""Tests for MI-APC, mfDCA, PLM-DCA zero-sum gauge invariance, and 3D contact recovery."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.coevolution import (
    CoevolutionConfig,
    _zero_sum_gauge_frobenius,
    apply_apc_correction,
    compare_coevolution_methods,
    compute_clustering_weights,
    compute_coupling_centrality,
    compute_mfdca,
    compute_mi_apc,
    compute_plmdca,
)
from clade_fold.synthetic_benchmark import generate_helical_hairpin_fold, simulate_potts_msa
from clade_fold.types import MultipleSequenceAlignment


def test_zero_sum_gauge_invariance() -> None:
    rng = np.random.default_rng(123)
    J = rng.normal(0.0, 1.0, size=(6, 6, 21, 21))
    frob1 = _zero_sum_gauge_frobenius(J)

    # Shift J_{ij}(a, b) -> J_{ij}(a, b) + u_i(a) + v_j(b) (gauge transformation)
    u = rng.normal(0.0, 5.0, size=(6, 6, 21, 1))
    v = rng.normal(0.0, 5.0, size=(6, 6, 1, 21))
    J_shifted = J + u + v
    frob2 = _zero_sum_gauge_frobenius(J_shifted)
    assert np.allclose(frob1, frob2, atol=1e-10)


def test_clustering_weights_and_apc_symmetry() -> None:
    msa = MultipleSequenceAlignment(
        names=["a1", "a2", "b1"],
        sequences=["ACDEFGHIKL", "ACDEFGHIKL", "WYVTSRQPNM"],
    )
    w, m_eff = compute_clustering_weights(msa.matrix, identity_threshold=0.8)
    assert w[0] == pytest.approx(0.5)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(1.0)
    assert m_eff == pytest.approx(2.0)

    raw = np.arange(100, dtype=np.float64).reshape(10, 10)
    apc = apply_apc_correction(raw, min_seq_sep=4)
    assert np.allclose(apc, apc.T)
    assert np.allclose(np.diag(apc), 0.0)
    assert apc[0, 2] == 0.0


def test_plmdca_and_mfdca_recover_3d_potts_contacts() -> None:
    struct = generate_helical_hairpin_fold(length=28, seed=19)
    coev_msa = simulate_potts_msa(
        struct, n_seqs=220, coupling_strength=2.2, burnin_sweeps=14, seed=21
    )
    cfg = CoevolutionConfig(min_seq_sep=6, plm_maxiter=50, top_k_pairs=30)
    results = compare_coevolution_methods(coev_msa, structure=struct, config=cfg)

    for method in ("MI-APC", "mfDCA", "PLM-DCA"):
        res = results[method]
        assert res.score_matrix.shape == (28, 28)
        assert np.allclose(res.score_matrix, res.score_matrix.T, atol=1e-10)
        assert res.precision_at_l5 is not None
        assert res.auroc is not None

    plm = results["PLM-DCA"]
    assert plm.precision_at_l5 is not None and plm.precision_at_l5 >= 0.50
    assert plm.auroc is not None and plm.auroc >= 0.70
    mfdca = results["mfDCA"]
    assert mfdca.precision_at_l5 is not None and mfdca.precision_at_l5 >= 0.60

    cent = compute_coupling_centrality(plm, min_seq_sep=6)
    assert cent.shape == (28,)
    assert float(np.sum(cent)) > 0.0

    # Individual entrypoints match
    mi_only = compute_mi_apc(coev_msa, cfg)
    mf_only = compute_mfdca(coev_msa, cfg)
    plm_only = compute_plmdca(coev_msa, cfg)
    assert np.allclose(mi_only.score_matrix, results["MI-APC"].score_matrix)
    assert np.allclose(mf_only.score_matrix, results["mfDCA"].score_matrix)
    assert np.allclose(plm_only.score_matrix, plm.score_matrix)
