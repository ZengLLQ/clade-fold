"""Tests for Kabsch, TM-score, GDT-TS, DSSP H-bonds, Ramachandran, and Shrake-Rupley SASA."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.structural_geometry import (
    analyze_structure,
    compute_backbone_dihedrals,
    compute_gdt_ts,
    compute_shrake_rupley_sasa,
    compute_tm_score,
    kabsch_superpose,
    summarize_dssp,
)
from clade_fold.synthetic_benchmark import (
    generate_beta_hairpin_fold,
    generate_helical_hairpin_fold,
)


def test_kabsch_rigid_transform_and_no_reflection() -> None:
    struct = generate_helical_hairpin_fold(length=24, seed=7)
    ca = struct.ca_coords

    # Apply proper SO(3) rotation + translation
    theta = 0.85
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    t_true = np.array([12.5, -7.2, 4.1], dtype=np.float64)
    mobile = (ca @ R_true.T) + t_true[None, :]

    aligned, R_est, _, rmsd = kabsch_superpose(mobile, ca)
    assert rmsd < 1e-10
    assert np.allclose(aligned, ca, atol=1e-10)
    assert float(np.linalg.det(R_est)) == pytest.approx(1.0, abs=1e-8)

    # Mirror image must yield positive RMSD and proper det(R) = +1
    mirrored = ca.copy()
    mirrored[:, 2] *= -1.0
    _, R_mir, _, rmsd_mir = kabsch_superpose(mirrored, ca)
    assert float(np.linalg.det(R_mir)) == pytest.approx(1.0, abs=1e-8)
    assert rmsd_mir > 0.5

    assert compute_tm_score(mobile, ca) == pytest.approx(1.0, abs=1e-6)
    assert compute_gdt_ts(mobile, ca) == pytest.approx(1.0, abs=1e-6)


def test_dssp_and_ramachandran_on_helix_and_beta_folds() -> None:
    helix_fold = generate_helical_hairpin_fold(length=30, seed=10)
    prof_h = analyze_structure(helix_fold)
    counts_h = summarize_dssp(prof_h.dssp_codes)
    assert counts_h["H"] >= 14
    assert len(prof_h.hydrogen_bonds) >= 8
    assert prof_h.ramachandran_zones.count("alpha_R") >= 16

    phi, psi, omega = compute_backbone_dihedrals(helix_fold)
    # Trans peptide bonds omega near +/- 180 deg
    valid_om = omega[~np.isnan(omega)]
    assert np.all(np.abs(valid_om) > 170.0)
    assert np.nanmean(phi) < 0.0

    beta_fold = generate_beta_hairpin_fold(length=28, seed=14)
    prof_b = analyze_structure(beta_fold)
    assert prof_b.ramachandran_zones.count("beta") >= 10


def test_shrake_rupley_sasa_termini_more_exposed_than_core() -> None:
    struct = generate_helical_hairpin_fold(length=36, seed=3)
    sasa, rsa, buried, solv_e = compute_shrake_rupley_sasa(struct, n_sphere_points=92)
    assert sasa.shape == (36,)
    assert np.all(sasa >= 0.0)
    assert float(np.mean(sasa)) > 20.0
    assert np.all((rsa >= 0.0) & (rsa <= 1.5))
    assert np.isfinite(solv_e)
    # Terminal residues should have higher mean RSA than interior residues
    term_rsa = 0.5 * (rsa[0] + rsa[-1])
    core_rsa = float(np.mean(rsa[5:-5]))
    assert term_rsa > core_rsa
    assert buried.dtype == np.bool_
