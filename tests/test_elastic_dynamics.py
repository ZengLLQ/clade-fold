"""Tests for GNM Kirchhoff, ANM Hessian 6 zero modes, deformation overlap, and evo-dynamics."""

from __future__ import annotations

import numpy as np

from clade_fold.elastic_dynamics import (
    ElasticNetworkConfig,
    analyze_evolution_dynamics_coupling,
    build_anm_hessian,
    build_gnm_kirchhoff,
    compute_deformation_overlap,
    run_elastic_network_analysis,
)
from clade_fold.io_formats import compute_virtual_cb
from clade_fold.structural_geometry import compute_shrake_rupley_sasa
from clade_fold.synthetic_benchmark import generate_helical_hairpin_fold
from clade_fold.types import ProteinStructure, ResidueRecord


def test_gnm_kirchhoff_and_anm_zero_mode_invariants() -> None:
    struct = generate_helical_hairpin_fold(length=28, seed=9)
    ca = struct.ca_coords

    Gamma = build_gnm_kirchhoff(ca, cutoff=7.0, gamma=1.0)
    assert np.allclose(Gamma, Gamma.T)
    assert np.allclose(np.sum(Gamma, axis=1), 0.0, atol=1e-10)
    g_evals = np.linalg.eigvalsh(Gamma)
    assert abs(float(g_evals[0])) < 1e-10
    assert float(g_evals[1]) > 1e-5

    H = build_anm_hessian(ca, cutoff=13.0, gamma=1.0)
    assert np.allclose(H, H.T)
    a_evals = np.linalg.eigvalsh(H)
    # Exactly 6 rigid-body zero modes (3 translations + 3 rotations)
    assert np.all(np.abs(a_evals[:6]) < 1e-8)
    assert float(a_evals[6]) > 1e-5


def test_elastic_network_bfactor_correlation_and_deformation_overlap() -> None:
    struct = generate_helical_hairpin_fold(length=30, seed=15)
    nma = run_elastic_network_analysis(struct, ElasticNetworkConfig())
    assert nma.bfactor_pearson is not None and nma.bfactor_pearson > 0.85
    assert nma.cross_correlation.shape == (30, 30)
    assert np.allclose(np.diag(nma.cross_correlation), 1.0, atol=1e-8)
    assert np.all((nma.cross_correlation >= -1.0001) & (nma.cross_correlation <= 1.0001))
    assert np.all((nma.collectivity > 0.0) & (nma.collectivity <= 1.0))
    assert len(nma.hinge_residues) >= 1

    # Perturb apo structure along slowest ANM mode and verify overlap ~ 1.0
    slow_mode = nma.anm_eigenvectors[:, 0].reshape(struct.length, 3)
    holo_res = []
    for i, r in enumerate(struct.residues):
        shift = 1.5 * slow_mode[i]
        new_n = r.n + shift
        new_ca = r.ca + shift
        new_c = r.c + shift
        new_o = r.o + shift
        holo_res.append(
            ResidueRecord(
                index=r.index,
                resseq=r.resseq,
                chain_id=r.chain_id,
                aa=r.aa,
                n=new_n,
                ca=new_ca,
                c=new_c,
                o=new_o,
                cb=compute_virtual_cb(new_n, new_ca, new_c),
                b_factor=r.b_factor,
            )
        )
    holo = ProteinStructure(name="holo", residues=holo_res)
    ov = compute_deformation_overlap(struct, holo, n_modes=6)
    assert float(ov["cumulative_overlap"]) > 0.90  # type: ignore[arg-type]

    _, rsa, _, _ = compute_shrake_rupley_sasa(struct)
    rng = np.random.default_rng(1)
    coupling = analyze_evolution_dynamics_coupling(
        nma,
        coupling_centrality=rng.uniform(0.0, 1.0, size=struct.length),
        conservation_scores=rng.uniform(0.2, 0.9, size=struct.length),
        rsa=rsa,
    )
    assert -1.0 <= coupling["spearman_rsa_vs_bfactor"] <= 1.0
