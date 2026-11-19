"""3D protein geometry: Kabsch superposition, TM-score, DSSP, Ramachandran, and SASA."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from .types import (
    BACKBONE_VDW_RADII,
    EISENBERG_HYDROPHOBICITY,
    FloatArray,
    MAX_SASA_TIEN,
    ProteinStructure,
    RESIDUE_EFFECTIVE_CB_RADIUS,
    SecondaryStructureProfile,
)


def kabsch_superpose(
    mobile: FloatArray, target: FloatArray
) -> Tuple[FloatArray, FloatArray, FloatArray, float]:
    """Compute optimal rigid-body superposition of ``mobile`` onto ``target`` (shape (N, 3)).

    Solves min_{R in SO(3), t} || (mobile @ R.T + t) - target ||_F^2 via SVD,
    enforcing det(R) = +1 to preclude improper mirror reflections.

    Returns:
        (superposed_mobile, rotation_matrix, translation_vector, rmsd_angstrom)
    """
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected identical (N, 3) arrays, got {mobile.shape} and {target.shape}")
    if mobile.shape[0] < 3:
        raise ValueError("Kabsch superposition requires at least 3 points")

    c_mob = np.mean(mobile, axis=0)
    c_tgt = np.mean(target, axis=0)
    P = mobile - c_mob
    Q = target - c_tgt

    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = 1.0 if float(np.linalg.det(Vt.T @ U.T)) >= 0.0 else -1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = c_tgt - (c_mob @ R.T)
    aligned = (mobile @ R.T) + t[None, :]
    diff = aligned - target
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return aligned, R, t, rmsd


def compute_tm_score(mobile_ca: FloatArray, target_ca: FloatArray) -> float:
    """Compute Zhang-Skolnick TM-score in (0, 1] using iterative Kabsch refinement.

    d_0(L) = max(0.5, 1.24 * (L - 15)^{1/3} - 1.8).
    """
    L = target_ca.shape[0]
    if mobile_ca.shape[0] != L:
        raise ValueError("mobile_ca and target_ca must have identical residue count")
    if L < 5:
        _, _, _, rmsd = kabsch_superpose(mobile_ca, target_ca)
        return float(1.0 / (1.0 + (rmsd / 0.5) ** 2))

    d0 = max(0.5, 1.24 * ((max(L, 16) - 15) ** (1.0 / 3.0)) - 1.8)
    d0_sq = d0 * d0

    best_tm = 0.0
    # Seed windows at multiple scales (full, L//2, L//4) for robust global superposition
    windows = [L, max(4, L // 2), max(4, L // 4)]
    for win in windows:
        step = max(1, win // 2)
        for start in range(0, L - win + 1, step):
            idx = np.arange(start, start + win)
            _, _, _, _ = kabsch_superpose(mobile_ca[idx], target_ca[idx])
            mob_sub = mobile_ca[idx]
            tgt_sub = target_ca[idx]
            aligned, R, t, _ = kabsch_superpose(mob_sub, tgt_sub)
            full_aligned = (mobile_ca @ R.T) + t[None, :]
            for _ in range(3):
                d2 = np.sum((full_aligned - target_ca) ** 2, axis=1)
                tm = float(np.mean(1.0 / (1.0 + d2 / d0_sq)))
                if tm > best_tm:
                    best_tm = tm
                # Re-weight subset with d < 2*d0
                mask = d2 < (4.0 * d0_sq)
                if int(np.sum(mask)) >= 4:
                    _, R, t, _ = kabsch_superpose(mobile_ca[mask], target_ca[mask])
                    full_aligned = (mobile_ca @ R.T) + t[None, :]
    return best_tm


def compute_gdt_ts(mobile_ca: FloatArray, target_ca: FloatArray) -> float:
    """Compute Global Distance Test Total Score (GDT-TS) across 1, 2, 4, 8 Angstrom cutoffs."""
    aligned, _, _, _ = kabsch_superpose(mobile_ca, target_ca)
    dists = np.sqrt(np.sum((aligned - target_ca) ** 2, axis=1))
    fracs = [float(np.mean(dists <= c)) for c in (1.0, 2.0, 4.0, 8.0)]
    return float(np.mean(fracs))


def compute_dihedral_deg(p0: FloatArray, p1: FloatArray, p2: FloatArray, p3: FloatArray) -> float:
    """Compute signed IUPAC torsion dihedral angle (degrees in [-180, 180]) for p0-p1-p2-p3."""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = float(np.linalg.norm(b1))
    if b1_norm < 1e-8:
        return 0.0
    b1_u = b1 / b1_norm
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1_u, v), w))
    return float(np.degrees(np.arctan2(y, x)))


def compute_backbone_dihedrals(
    structure: ProteinStructure,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Compute per-residue (phi, psi, omega) backbone dihedrals in degrees (NaN at termini)."""
    L = structure.length
    N = structure.n_coords
    CA = structure.ca_coords
    C = structure.c_coords

    phi = np.full(L, np.nan, dtype=np.float64)
    psi = np.full(L, np.nan, dtype=np.float64)
    omega = np.full(L, np.nan, dtype=np.float64)

    for i in range(L):
        if i > 0:
            phi[i] = compute_dihedral_deg(C[i - 1], N[i], CA[i], C[i])
            omega[i] = compute_dihedral_deg(CA[i - 1], C[i - 1], N[i], CA[i])
        if i < L - 1:
            psi[i] = compute_dihedral_deg(N[i], CA[i], C[i], N[i + 1])
    return phi, psi, omega


# Gaussian mixture modes for Ramachandran density:
# (center_phi, center_psi, sig_phi, sig_psi, weight, label)
_RAMA_MODES: Tuple[Tuple[float, float, float, float, float, str], ...] = (
    (-63.0, -43.0, 18.0, 18.0, 0.42, "alpha_R"),   # Right-handed alpha helix
    (-120.0, 130.0, 26.0, 26.0, 0.34, "beta"),     # Beta-sheet / extended
    (-75.0, 145.0, 20.0, 22.0, 0.16, "ppII"),      # Polyproline II
    (58.0, 42.0, 18.0, 20.0, 0.08, "alpha_L"),     # Left-handed alpha helix
)


def _wrapped_angle_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def score_ramachandran(
    phi: FloatArray, psi: FloatArray
) -> Tuple[List[str], FloatArray]:
    """Classify each residue into Ramachandran zones and compute GMM log-likelihoods."""
    L = len(phi)
    zones: List[str] = []
    log_lik = np.zeros(L, dtype=np.float64)

    for i in range(L):
        ph = float(phi[i])
        ps = float(psi[i])
        if np.isnan(ph) or np.isnan(ps):
            zones.append("terminus")
            log_lik[i] = 0.0
            continue

        density = 0.0
        best_mode = "outlier"
        best_contrib = 0.0
        for mu_ph, mu_ps, s_ph, s_ps, weight, label in _RAMA_MODES:
            dph = _wrapped_angle_diff(ph, mu_ph) / s_ph
            dps = _wrapped_angle_diff(ps, mu_ps) / s_ps
            z2 = 0.5 * (dph * dph + dps * dps)
            contrib = weight * float(np.exp(-z2)) / (2.0 * np.pi * s_ph * s_ps)
            density += contrib
            if z2 < 4.5 and contrib > best_contrib:
                best_contrib = contrib
                best_mode = label

        zones.append(best_mode)
        log_lik[i] = float(np.log(max(1e-12, density)))

    return zones, log_lik


def assign_dssp_secondary_structure(
    structure: ProteinStructure, hb_cutoff: float = -0.5
) -> Tuple[str, List[Tuple[int, int, float]]]:
    """Assign DSSP-style secondary structure via Kabsch-Sander backbone H-bond electrostatics.

    Computes virtual amide protons H_i along the C=O bond orientation of residue i-1:
      E(i_donor -> j_acceptor) = 27.888 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) kcal/mol
    H-bonds with E < hb_cutoff (-0.5 kcal/mol) drive 4-turns (H), 3-turns (G), 5-turns (I),
    and parallel/antiparallel beta-bridges/sheets (E/B).
    """
    L = structure.length
    N = structure.n_coords
    CA = structure.ca_coords
    C = structure.c_coords
    O_pts = structure.o_coords

    # Construct virtual amide H_i for i >= 1 (except Proline)
    H_pos = np.full((L, 3), np.nan, dtype=np.float64)
    is_pro = [r.aa == "P" for r in structure.residues]
    for i in range(1, L):
        if is_pro[i]:
            continue
        co_vec = C[i - 1] - O_pts[i - 1]
        norm_co = float(np.linalg.norm(co_vec))
        if norm_co > 1e-6:
            H_pos[i] = N[i] + 1.01 * (co_vec / norm_co)

    hbonds: List[Tuple[int, int, float]] = []
    hb_matrix = np.zeros((L, L), dtype=bool)

    for donor in range(1, L):
        if is_pro[donor] or np.isnan(H_pos[donor, 0]):
            continue
        n_d = N[donor]
        h_d = H_pos[donor]
        for acc in range(L):
            if abs(donor - acc) < 3:
                continue
            # Fast CA pre-filter (< 9.5 Angstrom)
            if float(np.linalg.norm(CA[donor] - CA[acc])) > 9.5:
                continue
            c_a = C[acc]
            o_a = O_pts[acc]
            r_on = max(0.5, float(np.linalg.norm(o_a - n_d)))
            r_ch = max(0.5, float(np.linalg.norm(c_a - h_d)))
            r_oh = max(0.5, float(np.linalg.norm(o_a - h_d)))
            r_cn = max(0.5, float(np.linalg.norm(c_a - n_d)))
            energy = 27.888 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
            if energy < hb_cutoff:
                hb_matrix[donor, acc] = True
                hbonds.append((donor, acc, float(energy)))

    # Secondary structure codes priority: H > E > B > G > I > T > C
    ss = ["C"] * L

    # 1. Detect n-turns: H-bond from donor i+n -> acceptor i
    turn3 = np.array([i + 3 < L and hb_matrix[i + 3, i] for i in range(L)], dtype=bool)
    turn4 = np.array([i + 4 < L and hb_matrix[i + 4, i] for i in range(L)], dtype=bool)
    turn5 = np.array([i + 5 < L and hb_matrix[i + 5, i] for i in range(L)], dtype=bool)

    # Mark turns
    for i in range(L):
        if turn3[i]:
            for k in range(i + 1, min(L, i + 3)):
                if ss[k] == "C":
                    ss[k] = "T"
        if turn4[i]:
            for k in range(i + 1, min(L, i + 4)):
                if ss[k] == "C":
                    ss[k] = "T"

    # 3_10 helix (G): two consecutive 3-turns
    for i in range(L - 1):
        if turn3[i] and turn3[i + 1]:
            for k in range(i + 1, min(L, i + 4)):
                ss[k] = "G"

    # Pi-helix (I): two consecutive 5-turns
    for i in range(L - 1):
        if turn5[i] and turn5[i + 1]:
            for k in range(i + 1, min(L, i + 6)):
                ss[k] = "I"

    # Beta-bridges and Beta-sheets (B / E)
    bridge = np.zeros(L, dtype=bool)
    for i in range(1, L - 1):
        for j in range(1, L - 1):
            if abs(i - j) < 3:
                continue
            parallel = (hb_matrix[i, j - 1] and hb_matrix[j + 1, i]) or (
                hb_matrix[j, i - 1] and hb_matrix[i + 1, j]
            )
            antiparallel = (hb_matrix[i, j] and hb_matrix[j, i]) or (
                hb_matrix[i + 1, j - 1] and hb_matrix[j + 1, i - 1]
            )
            if parallel or antiparallel:
                bridge[i] = True
                bridge[j] = True

    for i in range(L):
        if bridge[i]:
            has_adj = (i > 0 and bridge[i - 1]) or (i + 1 < L and bridge[i + 1])
            ss[i] = "E" if has_adj else "B"

    # Alpha-helix (H): highest priority when consecutive 4-turns occur
    for i in range(L - 1):
        if turn4[i] and turn4[i + 1]:
            for k in range(i + 1, min(L, i + 5)):
                ss[k] = "H"

    return "".join(ss), hbonds


def _fibonacci_sphere(n_points: int = 92) -> FloatArray:
    """Generate quasi-uniform unit sphere quadrature points via the Golden Spiral."""
    indices = np.arange(n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / float(n_points))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def compute_shrake_rupley_sasa(
    structure: ProteinStructure,
    probe_radius: float = 1.4,
    n_sphere_points: int = 92,
) -> Tuple[FloatArray, FloatArray, NDArray[np.bool_], float]:
    """Compute per-residue SASA (Angstrom^2), RSA, buried mask, and Eisenberg solvation energy.

    Models each residue by its 5 steric spheres (N, CA, C, O, and effective sidechain CB)
    with Golden-Spiral quadrature over probe-expanded van der Waals radii.
    """
    L = structure.length
    unit_pts = _fibonacci_sphere(n_sphere_points)

    # Flatten all 5 spheres per residue (5*L atoms total)
    centers: List[FloatArray] = []
    radii: List[float] = []
    owner_res: List[int] = []

    for idx, res in enumerate(structure.residues):
        cb_rad = RESIDUE_EFFECTIVE_CB_RADIUS.get(res.aa, 2.5)
        atom_specs = [
            (res.n, BACKBONE_VDW_RADII["N"] + probe_radius),
            (res.ca, BACKBONE_VDW_RADII["CA"] + probe_radius),
            (res.c, BACKBONE_VDW_RADII["C"] + probe_radius),
            (res.o, BACKBONE_VDW_RADII["O"] + probe_radius),
            (res.cb, cb_rad + probe_radius),
        ]
        for xyz, rad in atom_specs:
            centers.append(xyz)
            radii.append(rad)
            owner_res.append(idx)

    C_arr = np.vstack(centers)  # (M, 3)
    R_arr = np.array(radii, dtype=np.float64)  # (M,)
    M = len(R_arr)

    # Pairwise center distance matrix
    diff = C_arr[:, None, :] - C_arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # Two spheres i, j can occlude each other iff d(i, j) < R_i + R_j
    sum_r = R_arr[:, None] + R_arr[None, :]
    overlap = (dist < sum_r) & (np.arange(M)[:, None] != np.arange(M)[None, :])

    res_sasa = np.zeros(L, dtype=np.float64)
    for a_idx in range(M):
        nbrs = np.where(overlap[a_idx])[0]
        r_a = R_arr[a_idx]
        area_total = 4.0 * np.pi * r_a * r_a
        if nbrs.size == 0:
            res_sasa[owner_res[a_idx]] += area_total
            continue
        pts = C_arr[a_idx][None, :] + r_a * unit_pts  # (P, 3)
        nbr_c = C_arr[nbrs]  # (K, 3)
        nbr_r2 = (R_arr[nbrs] ** 2)[None, :]  # (1, K)
        # (P, K) squared distances
        d2 = np.sum((pts[:, None, :] - nbr_c[None, :, :]) ** 2, axis=-1)
        exposed = ~np.any(d2 < nbr_r2, axis=1)
        frac = float(np.mean(exposed))
        res_sasa[owner_res[a_idx]] += frac * area_total

    rsa = np.zeros(L, dtype=np.float64)
    solv_energy = 0.0
    for i, res in enumerate(structure.residues):
        max_s = MAX_SASA_TIEN.get(res.aa, 200.0)
        rsa_i = min(1.5, float(res_sasa[i]) / max_s)
        rsa[i] = rsa_i
        # Buried hydrophobic residues stabilize folding; exposed hydrophobic residues penalize
        hydro = EISENBERG_HYDROPHOBICITY.get(res.aa, 0.0)
        solv_energy += (rsa_i - 0.5) * hydro

    buried_mask = rsa < 0.20
    return res_sasa, rsa, buried_mask, float(solv_energy)


def analyze_structure(structure: ProteinStructure) -> SecondaryStructureProfile:
    """Run integrated DSSP, Ramachandran, and Shrake-Rupley SASA structural analysis."""
    phi, psi, omega = compute_backbone_dihedrals(structure)
    zones, rama_ll = score_ramachandran(phi, psi)
    dssp, hbonds = assign_dssp_secondary_structure(structure)
    sasa, rsa, buried_mask, solv_energy = compute_shrake_rupley_sasa(structure)
    return SecondaryStructureProfile(
        dssp_codes=dssp,
        phi_deg=phi,
        psi_deg=psi,
        omega_deg=omega,
        ramachandran_zones=zones,
        ramachandran_log_likelihood=rama_ll,
        sasa=sasa,
        rsa=rsa,
        buried_mask=buried_mask,
        solvation_energy=solv_energy,
        hydrogen_bonds=hbonds,
    )


def summarize_dssp(dssp_codes: str) -> Dict[str, int]:
    """Count secondary structure element occurrences across DSSP codes."""
    counts: Dict[str, int] = {"H": 0, "E": 0, "B": 0, "G": 0, "I": 0, "T": 0, "C": 0}
    for ch in dssp_codes:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
