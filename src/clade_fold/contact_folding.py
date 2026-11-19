"""Contact-restrained 3D C-alpha distance-geometry embedding and L-BFGS-B folding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from .structural_geometry import compute_dihedral_deg
from .types import ContactMapResult, FloatArray


@dataclass(frozen=True)
class FoldingConfig:
    """Energy weights and optimization bounds for contact-restrained C-alpha folding."""

    ca_bond_length: float = 3.80
    k_bond: float = 40.0
    k_local_ss: float = 12.0
    k_contact: float = 6.0
    k_clash: float = 25.0
    clash_cutoff: float = 3.60
    contact_upper: float = 8.00
    contact_lower: float = 4.20
    top_k_contacts: Optional[int] = None
    maxiter: int = 180
    n_restarts: int = 2
    seed: int = 42


@dataclass
class FoldingResult:
    """Reconstructed 3D C-alpha trace and potential energy decomposition."""

    ca_coords: FloatArray
    total_energy: float
    bond_energy: float
    local_ss_energy: float
    contact_energy: float
    clash_energy: float
    helix_handedness_score: float


def mds_distance_embedding(dist_matrix: FloatArray) -> FloatArray:
    """Compute classical Multidimensional Scaling (Torgerson) 3D embedding from distances."""
    L = dist_matrix.shape[0]
    D2 = dist_matrix * dist_matrix
    row_mean = np.mean(D2, axis=1, keepdims=True)
    col_mean = np.mean(D2, axis=0, keepdims=True)
    grand_mean = float(np.mean(D2))
    G = -0.5 * (D2 - row_mean - col_mean + grand_mean)
    G = 0.5 * (G + G.T)

    evals, evecs = np.linalg.eigh(G)
    top3 = np.argsort(evals)[::-1][:3]
    lam = np.maximum(0.0, evals[top3])
    coords = np.asarray(evecs[:, top3] * np.sqrt(lam)[None, :], dtype=np.float64)
    if coords.shape != (L, 3):
        out = np.zeros((L, 3), dtype=np.float64)
        out[:, : coords.shape[1]] = coords
        return out
    return coords


def _build_initial_distance_guess(
    L: int,
    contacts: List[Tuple[int, int, float]],
    dssp_codes: str,
    ca_bond: float = 3.80,
) -> FloatArray:
    """Build shortest-path graph-metric distance matrix from local SS and DCA contacts."""
    D = np.full((L, L), 1e4, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for i in range(L - 1):
        D[i, i + 1] = ca_bond
        D[i + 1, i] = ca_bond
    for i in range(L - 2):
        code = dssp_codes[i + 1] if i + 1 < len(dssp_codes) else "C"
        d2 = 5.45 if code in ("H", "G", "I") else (6.65 if code in ("E", "B") else 5.95)
        D[i, i + 2] = d2
        D[i + 2, i] = d2
    for i in range(L - 3):
        code = dssp_codes[i + 1] if i + 1 < len(dssp_codes) else "C"
        d3 = 5.15 if code in ("H", "G", "I") else (9.70 if code in ("E", "B") else 7.50)
        D[i, i + 3] = d3
        D[i + 3, i] = d3
    for i in range(L - 4):
        if i + 4 < len(dssp_codes) and dssp_codes[i + 1:i + 4].count("H") >= 2:
            D[i, i + 4] = 6.20
            D[i + 4, i] = 6.20

    for i, j, _ in contacts:
        D[i, j] = min(D[i, j], 6.4)
        D[j, i] = D[i, j]

    # Floyd-Warshall all-pairs shortest path metric
    for k in range(L):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    return D


def _pair_harmonic_term(
    coords: FloatArray,
    grad: FloatArray,
    i: int,
    j: int,
    weight: float,
    d_min: float,
    d_max: float,
) -> float:
    """Accumulate flat-bottom harmonic restraint energy and exact gradient for pair (i, j)."""
    vec = coords[i] - coords[j]
    dist = float(np.sqrt(np.dot(vec, vec) + 1e-12))
    if dist < d_min:
        delta = dist - d_min
    elif dist > d_max:
        delta = dist - d_max
    else:
        return 0.0
    e = weight * delta * delta
    coeff = (2.0 * weight * delta) / dist
    g_vec = coeff * vec
    grad[i] += g_vec
    grad[j] -= g_vec
    return e


def _right_handed_helix_score(coords: FloatArray, dssp_codes: str) -> float:
    """Measure mean signed virtual CA dihedral in helical spans (+50 deg is right-handed)."""
    scores: List[float] = []
    L = coords.shape[0]
    for i in range(L - 3):
        seg = dssp_codes[i:i + 4] if i + 4 <= len(dssp_codes) else ""
        if seg.count("H") >= 3:
            ang = compute_dihedral_deg(coords[i], coords[i + 1], coords[i + 2], coords[i + 3])
            scores.append(ang)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def reconstruct_ca_fold(
    contact_result: ContactMapResult,
    dssp_codes: Optional[str] = None,
    config: Optional[FoldingConfig] = None,
) -> FoldingResult:
    """Reconstruct a 3D C-alpha backbone model from DCA contacts and secondary-structure restraints.

    Initializes 3D coordinates via Floyd-Warshall + classical MDS embedding, then
    minimizes a smooth analytical potential combining CA-CA bond lengths, local
    helix/strand i->i+2/i+3 geometry, flat-bottom DCA contact bounds, and steric
    clash repulsion using L-BFGS-B. Automatically resolves global mirror chirality.
    """
    cfg = config or FoldingConfig()
    L = contact_result.score_matrix.shape[0]
    ss = dssp_codes if dssp_codes is not None and len(dssp_codes) >= L else ("C" * L)

    k_top = cfg.top_k_contacts if cfg.top_k_contacts is not None else max(6, L // 2)
    # Keep positive-scoring top contacts
    pos_contacts = [(i, j, s) for i, j, s in contact_result.top_pairs if s > 0.0][:k_top]
    max_s = max((s for _, _, s in pos_contacts), default=1.0)
    norm_contacts = [(i, j, max(0.1, s / max(1e-8, max_s))) for i, j, s in pos_contacts]

    # Build local SS restraints: (i, j, weight, d_min, d_max)
    local_restraints: List[Tuple[int, int, float, float, float]] = []
    helix_quads: List[int] = []
    for i in range(L - 2):
        code = ss[i + 1]
        if code in ("H", "G", "I"):
            local_restraints.append((i, i + 2, cfg.k_local_ss, 5.25, 5.60))
        elif code in ("E", "B"):
            local_restraints.append((i, i + 2, cfg.k_local_ss, 6.20, 7.00))
    for i in range(L - 3):
        code = ss[i + 1]
        if code in ("H", "G", "I"):
            local_restraints.append((i, i + 3, cfg.k_local_ss, 4.90, 5.35))
            if ss[i:i + 4].count("H") >= 3:
                helix_quads.append(i)
        elif code in ("E", "B"):
            local_restraints.append((i, i + 3, 0.5 * cfg.k_local_ss, 9.00, 10.40))
    for i in range(L - 4):
        if ss[i + 1:i + 4].count("H") >= 2:
            local_restraints.append((i, i + 4, cfg.k_local_ss, 5.95, 6.45))

    d_init = _build_initial_distance_guess(L, norm_contacts, ss, cfg.ca_bond_length)
    x_mds = mds_distance_embedding(d_init)

    def _eval_energy(x_flat: FloatArray) -> Tuple[float, FloatArray, Dict[str, float]]:
        coords = x_flat.reshape(L, 3)
        grad = np.zeros_like(coords)
        e_bond = 0.0
        e_ss = 0.0
        e_cont = 0.0
        e_clash = 0.0

        # 1. Sequential CA-CA virtual bonds
        for i in range(L - 1):
            e_bond += _pair_harmonic_term(
                coords, grad, i, i + 1, cfg.k_bond, cfg.ca_bond_length, cfg.ca_bond_length
            )

        # 2. Local secondary structure geometry + right-handed helix chiral triple product
        for i, j, w, d_lo, d_hi in local_restraints:
            e_ss += _pair_harmonic_term(coords, grad, i, j, w, d_lo, d_hi)

        target_vol = 20.0
        k_chiral = 0.25
        for i in helix_quads:
            v1 = coords[i + 1] - coords[i]
            v2 = coords[i + 2] - coords[i + 1]
            v3 = coords[i + 3] - coords[i + 2]
            c12 = np.cross(v1, v2)
            vol = float(np.dot(c12, v3))
            if vol < target_vol:
                dv = vol - target_vol
                e_ss += k_chiral * dv * dv
                scale = 2.0 * k_chiral * dv
                # Exact gradients of triple product det([v1, v2, v3])
                g_v1 = scale * np.cross(v2, v3)
                g_v2 = scale * np.cross(v3, v1)
                g_v3 = scale * c12
                grad[i] -= g_v1
                grad[i + 1] += g_v1 - g_v2
                grad[i + 2] += g_v2 - g_v3
                grad[i + 3] += g_v3

        # 3. DCA contact upper/lower bounds
        for i, j, w_norm in norm_contacts:
            e_cont += _pair_harmonic_term(
                coords, grad, i, j, cfg.k_contact * w_norm, cfg.contact_lower, cfg.contact_upper
            )

        # 4. Vectorized non-bonded steric clash repulsion for j >= i + 2
        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1) + 1e-12
        dist = np.sqrt(dist2)
        idx = np.arange(L)
        non_bonded = (idx[None, :] - idx[:, None]) >= 2
        clash_mask = non_bonded & (dist < cfg.clash_cutoff)
        if np.any(clash_mask):
            ii, jj = np.where(clash_mask)
            for a, b in zip(ii.tolist(), jj.tolist()):
                e_clash += _pair_harmonic_term(
                    coords, grad, int(a), int(b), cfg.k_clash, cfg.clash_cutoff, 1e6
                )

        tot = e_bond + e_ss + e_cont + e_clash
        breakdown = {
            "bond": e_bond,
            "local_ss": e_ss,
            "contact": e_cont,
            "clash": e_clash,
        }
        return tot, grad.ravel(), breakdown

    def _obj(x_flat: FloatArray) -> Tuple[float, FloatArray]:
        tot, g, _ = _eval_energy(x_flat)
        return tot, g

    rng = np.random.default_rng(cfg.seed)
    best_x = x_mds.ravel()
    best_e = np.inf

    for restart in range(max(1, cfg.n_restarts)):
        x0 = x_mds.copy()
        if restart > 0:
            x0 += rng.normal(0.0, 1.2, size=x0.shape)
        res = minimize(
            _obj,
            x0.ravel(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": cfg.maxiter, "ftol": 1e-7},
        )
        if float(res.fun) < best_e:
            best_e = float(res.fun)
            best_x = res.x.copy()

    final_coords = best_x.reshape(L, 3)
    # Center at origin
    final_coords -= np.mean(final_coords, axis=0, keepdims=True)
    # Resolve reflection chirality for right-handed alpha-helices (+50 deg virtual CA dihedral)
    hand = _right_handed_helix_score(final_coords, ss)
    if hand < -10.0:
        final_coords[:, 2] *= -1.0
        hand = _right_handed_helix_score(final_coords, ss)

    tot_e, _, parts = _eval_energy(final_coords.ravel())
    return FoldingResult(
        ca_coords=final_coords,
        total_energy=tot_e,
        bond_energy=parts["bond"],
        local_ss_energy=parts["local_ss"],
        contact_energy=parts["contact"],
        clash_energy=parts["clash"],
        helix_handedness_score=hand,
    )
