"""Gaussian & Anisotropic Elastic Network Models (GNM/ANM) and evolution-dynamics coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .structural_geometry import kabsch_superpose
from .types import FloatArray, NormalModeResult, ProteinStructure


@dataclass(frozen=True)
class ElasticNetworkConfig:
    """Cutoff and spring constants for GNM and ANM normal-mode analysis."""

    gnm_cutoff: float = 7.0
    anm_cutoff: float = 13.0
    spring_gamma: float = 1.0
    n_hinge_modes: int = 2
    hinge_quantile: float = 0.15


def build_gnm_kirchhoff(
    ca_coords: FloatArray, cutoff: float = 7.0, gamma: float = 1.0
) -> FloatArray:
    """Construct the N x N Gaussian Network Model Kirchhoff contact matrix Gamma."""
    N = ca_coords.shape[0]
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    adj = ((dist <= cutoff) & (dist > 1e-6)).astype(np.float64)
    # Ensure chain connectivity along backbone i <-> i+1 even in extended loops
    for i in range(N - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    Gamma = np.asarray(-gamma * adj, dtype=np.float64)
    np.fill_diagonal(Gamma, gamma * np.sum(adj, axis=1))
    return Gamma


def build_anm_hessian(
    ca_coords: FloatArray, cutoff: float = 13.0, gamma: float = 1.0
) -> FloatArray:
    """Construct the 3N x 3N Anisotropic Network Model (ANM) Hessian matrix H."""
    N = ca_coords.shape[0]
    H = np.zeros((3 * N, 3 * N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            rij = ca_coords[j] - ca_coords[i]
            d2 = float(np.dot(rij, rij))
            d = float(np.sqrt(d2))
            if d < 1e-6:
                continue
            if d <= cutoff or abs(i - j) == 1:
                block = -gamma * np.outer(rij, rij) / d2
                si = slice(3 * i, 3 * i + 3)
                sj = slice(3 * j, 3 * j + 3)
                H[si, sj] = block
                H[sj, si] = block
                H[si, si] -= block
                H[sj, sj] -= block
    return 0.5 * (H + H.T)


def _mode_collectivity(eigvecs_3n: FloatArray, n_residues: int) -> FloatArray:
    """Compute Bruschweiler (1995) collectivity kappa_m in [1/N, 1] for 3N modes."""
    n_modes = eigvecs_3n.shape[1]
    coll = np.zeros(n_modes, dtype=np.float64)
    for m in range(n_modes):
        disp = eigvecs_3n[:, m].reshape(n_residues, 3)
        amp2 = np.sum(disp * disp, axis=1)
        tot = float(np.sum(amp2))
        if tot <= 1e-12:
            coll[m] = 0.0
            continue
        p = amp2 / tot
        nz = p[p > 1e-12]
        entropy = -float(np.sum(nz * np.log(nz)))
        coll[m] = float(np.exp(entropy)) / float(n_residues)
    return coll


def run_elastic_network_analysis(
    structure: ProteinStructure, config: Optional[ElasticNetworkConfig] = None
) -> NormalModeResult:
    """Run coupled GNM and ANM normal-mode analysis on a ProteinStructure.

    Computes:
      - GNM eigenvalues/modes, theoretical B-factors, experimental B-factor correlation,
        and slow-mode hinge residues.
      - ANM 3N Hessian eigendecomposition (discarding 6 rigid-body zero modes),
        residue MSF, cross-correlation matrix C_{ij}, and mode collectivity.
    """
    cfg = config or ElasticNetworkConfig()
    ca = structure.ca_coords
    N = ca.shape[0]
    if N < 4:
        raise ValueError("Elastic network analysis requires at least 4 residues")

    # 1. GNM
    Gamma = build_gnm_kirchhoff(ca, cutoff=cfg.gnm_cutoff, gamma=cfg.spring_gamma)
    g_evals, g_evecs = np.linalg.eigh(Gamma)
    # Skip trivial zero mode (index 0)
    pos_g = g_evals > 1e-8
    inv_lam = np.where(pos_g, 1.0 / np.maximum(1e-8, g_evals), 0.0)
    gamma_pinv_diag = np.sum((g_evecs * g_evecs) * inv_lam[None, :], axis=1)

    exp_b = structure.b_factors
    # Scale GNM diagonal to match experimental B-factor mean when available
    if float(np.std(exp_b)) > 1e-6:
        scale = float(np.mean(exp_b)) / max(1e-12, float(np.mean(gamma_pinv_diag)))
        gnm_b = gamma_pinv_diag * scale
        r_val, _ = pearsonr(gnm_b, exp_b)
        b_corr: Optional[float] = float(r_val) if np.isfinite(r_val) else None
    else:
        gnm_b = gamma_pinv_diag * (8.0 * np.pi ** 2 / 3.0)
        b_corr = None

    # Hinge residues: local minima of weighted slow-mode mobility
    non_zero_idx = np.where(pos_g)[0]
    slow_idx = non_zero_idx[: max(1, min(cfg.n_hinge_modes, len(non_zero_idx)))]
    slow_mob = np.sum(
        (g_evecs[:, slow_idx] ** 2) / np.maximum(1e-8, g_evals[slow_idx])[None, :],
        axis=1,
    )
    thresh = float(np.quantile(slow_mob, cfg.hinge_quantile))
    hinges: List[int] = [int(i) for i in range(N) if slow_mob[i] <= thresh]

    # 2. ANM
    H = build_anm_hessian(ca, cutoff=cfg.anm_cutoff, gamma=cfg.spring_gamma)
    a_evals, a_evecs = np.linalg.eigh(H)
    # First 6 eigenvalues correspond to 3 translations + 3 rotations
    valid_a = np.where(a_evals > 1e-7)[0]
    if valid_a.size == 0:
        valid_a = np.arange(min(6, 3 * N), 3 * N)

    v_vecs = a_evecs[:, valid_a]
    v_vals = a_evals[valid_a]
    inv_a = 1.0 / v_vals
    # Covariance matrix C = H^+ = V diag(1/lambda) V^T
    cov_3n = (v_vecs * inv_a[None, :]) @ v_vecs.T

    anm_msf = np.zeros(N, dtype=np.float64)
    for i in range(N):
        block_ii = cov_3n[3 * i: 3 * i + 3, 3 * i: 3 * i + 3]
        anm_msf[i] = float(np.trace(block_ii))

    cross_corr = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            denom = float(np.sqrt(max(1e-12, anm_msf[i] * anm_msf[j])))
            tr_ij = float(np.trace(cov_3n[3 * i: 3 * i + 3, 3 * j: 3 * j + 3]))
            val = float(np.clip(tr_ij / denom, -1.0, 1.0))
            cross_corr[i, j] = val
            cross_corr[j, i] = val

    collectivity = _mode_collectivity(v_vecs, N)

    return NormalModeResult(
        gnm_eigenvalues=g_evals,
        gnm_eigenvectors=g_evecs,
        gnm_bfactors=gnm_b,
        bfactor_pearson=b_corr,
        hinge_residues=hinges,
        anm_eigenvalues=v_vals,
        anm_eigenvectors=v_vecs,
        anm_msf=anm_msf,
        cross_correlation=cross_corr,
        collectivity=collectivity,
    )


def compute_deformation_overlap(
    structure_apo: ProteinStructure,
    structure_holo: ProteinStructure,
    n_modes: int = 10,
    config: Optional[ElasticNetworkConfig] = None,
) -> Dict[str, object]:
    """Compute directional overlap between ANM normal modes and apo->holo conformational change."""
    ca_apo = structure_apo.ca_coords
    ca_holo = structure_holo.ca_coords
    if ca_apo.shape != ca_holo.shape:
        raise ValueError("Apo and holo structures must have identical residue count")

    holo_aligned, _, _, rmsd = kabsch_superpose(ca_holo, ca_apo)
    delta_x = (holo_aligned - ca_apo).ravel()
    norm_dx = float(np.linalg.norm(delta_x))
    if norm_dx < 1e-8:
        return {"rmsd": rmsd, "max_single_overlap": 1.0, "cumulative_overlap": 1.0, "overlaps": []}

    nma = run_elastic_network_analysis(structure_apo, config)
    k = min(n_modes, nma.anm_eigenvectors.shape[1])
    overlaps: List[float] = []
    for m in range(k):
        vm = nma.anm_eigenvectors[:, m]
        ov = abs(float(np.dot(delta_x, vm))) / (norm_dx * max(1e-12, float(np.linalg.norm(vm))))
        overlaps.append(ov)

    cum_ov = float(np.sqrt(min(1.0, sum(o * o for o in overlaps))))
    return {
        "rmsd": rmsd,
        "max_single_overlap": max(overlaps) if overlaps else 0.0,
        "cumulative_overlap": cum_ov,
        "overlaps": overlaps,
    }


def analyze_evolution_dynamics_coupling(
    nma_result: NormalModeResult,
    coupling_centrality: FloatArray,
    conservation_scores: FloatArray,
    rsa: FloatArray,
) -> Dict[str, float]:
    """Quantify coupling between co-variation, conservation, burial, and normal-mode dynamics."""
    L = min(
        len(nma_result.gnm_bfactors),
        len(coupling_centrality),
        len(conservation_scores),
        len(rsa),
    )
    bf = nma_result.gnm_bfactors[:L]
    cent = coupling_centrality[:L]
    cons = conservation_scores[:L]
    rsa_s = rsa[:L]

    def _safe_spearman(x: FloatArray, y: FloatArray) -> float:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        rho, _ = spearmanr(x, y)
        return float(rho) if np.isfinite(rho) else 0.0

    hinge_set = set(nma_result.hinge_residues)
    hinge_mask = np.array([i in hinge_set for i in range(L)], dtype=bool)
    hinge_cons = float(np.mean(cons[hinge_mask])) if np.any(hinge_mask) else float(np.mean(cons))
    non_hinge_cons = (
        float(np.mean(cons[~hinge_mask])) if np.any(~hinge_mask) else float(np.mean(cons))
    )

    return {
        "spearman_centrality_vs_bfactor": _safe_spearman(cent, bf),
        "spearman_conservation_vs_bfactor": _safe_spearman(cons, bf),
        "spearman_rsa_vs_bfactor": _safe_spearman(rsa_s, bf),
        "spearman_centrality_vs_rsa": _safe_spearman(cent, rsa_s),
        "hinge_mean_conservation": hinge_cons,
        "non_hinge_mean_conservation": non_hinge_cons,
    }
