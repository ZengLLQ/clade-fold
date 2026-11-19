"""Tests for MDS distance-geometry embedding and contact-restrained 3D folding."""

from __future__ import annotations

import numpy as np

from clade_fold.coevolution import CoevolutionConfig, compute_plmdca
from clade_fold.contact_folding import FoldingConfig, mds_distance_embedding, reconstruct_ca_fold
from clade_fold.structural_geometry import analyze_structure, compute_tm_score
from clade_fold.synthetic_benchmark import generate_helical_hairpin_fold, simulate_potts_msa


def test_mds_embedding_exact_isometry() -> None:
    struct = generate_helical_hairpin_fold(length=22, seed=4)
    dist = struct.ca_distance_matrix()
    emb = mds_distance_embedding(dist)
    assert emb.shape == (22, 3)
    flipped = emb.copy()
    flipped[:, 2] *= -1.0
    tm_orig = compute_tm_score(emb, struct.ca_coords)
    tm_flip = compute_tm_score(flipped, struct.ca_coords)
    assert max(tm_orig, tm_flip) > 0.98


def test_contact_restrained_folding_achieves_fold_topology() -> None:
    struct = generate_helical_hairpin_fold(length=28, seed=19)
    prof = analyze_structure(struct)
    coev_msa = simulate_potts_msa(struct, n_seqs=220, coupling_strength=2.2, seed=21)
    contact_res = compute_plmdca(coev_msa, CoevolutionConfig(plm_maxiter=45))

    fold_res = reconstruct_ca_fold(
        contact_res,
        dssp_codes=prof.dssp_codes,
        config=FoldingConfig(maxiter=180, n_restarts=2, seed=42),
    )
    assert fold_res.ca_coords.shape == (28, 3)
    assert fold_res.bond_energy < 10.0
    assert fold_res.clash_energy < 2.0
    assert fold_res.helix_handedness_score > 30.0

    # Check sequential CA-CA bond lengths near 3.8 Angstrom
    diffs = np.diff(fold_res.ca_coords, axis=0)
    blens = np.sqrt(np.sum(diffs * diffs, axis=1))
    assert float(np.mean(np.abs(blens - 3.8))) < 0.25

    tm = compute_tm_score(fold_res.ca_coords, struct.ca_coords)
    assert tm >= 0.30
