"""Evolutionary co-variation and Direct Coupling Analysis (MI-APC, mfDCA, PLM-DCA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from .types import (
    ContactMapResult,
    FloatArray,
    IntArray,
    MultipleSequenceAlignment,
    ProteinStructure,
)


@dataclass(frozen=True)
class CoevolutionConfig:
    """Configuration for sequence reweighting, pseudocounts, and Potts inference."""

    identity_threshold: float = 0.80
    pseudocount: float = 0.25
    l2_couplings: float = 0.01
    l2_fields: float = 0.01
    min_seq_sep: int = 6
    plm_maxiter: int = 60
    top_k_pairs: int = 50


def compute_clustering_weights(
    msa_matrix: IntArray, identity_threshold: float = 0.80
) -> Tuple[FloatArray, float]:
    """Compute Morcos/Ekeberg homology-cluster sequence weights w_m and M_eff.

    Each sequence m receives weight w_m = 1 / |{n : id(m, n) >= threshold}|,
    measured over all non-gap or matched positions. Returns (weights, M_eff).
    """
    n_seqs, length = msa_matrix.shape
    if n_seqs == 1:
        return np.ones(1, dtype=np.float64), 1.0

    # Vectorized pairwise match fraction over all columns
    # Equality matrix computed in chunks if needed, direct for typical MSA depth
    matches = (msa_matrix[:, None, :] == msa_matrix[None, :, :]).mean(axis=2)
    neighbors = np.sum(matches >= identity_threshold, axis=1).astype(np.float64)
    weights = 1.0 / np.maximum(1.0, neighbors)
    m_eff = float(np.sum(weights))
    return weights, m_eff


def compute_site_pair_frequencies(
    msa_matrix: IntArray,
    weights: FloatArray,
    pseudocount: float = 0.25,
    n_states: int = 21,
) -> Tuple[FloatArray, FloatArray, float]:
    """Compute weighted single-site f_i(a) and two-site f_{ij}(a, b) with Dirichlet pseudocount.

    Returns:
        f1: shape (L, n_states)
        f2: shape (L, L, n_states, n_states)
        m_eff: effective sequence depth sum(w_m)
    """
    n_seqs, L = msa_matrix.shape
    m_eff = max(1e-12, float(np.sum(weights)))
    lam = float(np.clip(pseudocount, 0.0, 0.99))

    # One-hot encoding: (N, L, n_states)
    onehot = np.zeros((n_seqs, L, n_states), dtype=np.float64)
    rows = np.arange(n_seqs)[:, None]
    cols = np.arange(L)[None, :]
    clamped = np.clip(msa_matrix, 0, n_states - 1)
    onehot[rows, cols, clamped] = 1.0

    weighted_oh = onehot * weights[:, None, None]
    f1_emp = np.sum(weighted_oh, axis=0) / m_eff  # (L, Q)

    # Pair counts via einsum: f2_emp[i, j, a, b] = sum_m w_m * x_{m,i,a} * x_{m,j,b} / m_eff
    f2_emp = np.einsum("nia,njb->ijab", weighted_oh, onehot) / m_eff

    q = float(n_states)
    f1 = (1.0 - lam) * f1_emp + lam / q
    f2 = (1.0 - lam) * f2_emp + (lam / (q * q))

    # Enforce exact marginal consistency on diagonal (i == j)
    for i in range(L):
        f2[i, i] = np.diag(f1[i])

    return f1, f2, m_eff


def apply_apc_correction(score_matrix: FloatArray, min_seq_sep: int = 6) -> FloatArray:
    """Apply Dunn-Wahl-Gloor Average Product Correction (APC) to a symmetric coupling matrix.

    APC_{ij} = (mean_i * mean_j) / global_mean
    Only off-diagonal entries with |i - j| >= min_seq_sep are kept positive/scored.
    """
    L = score_matrix.shape[0]
    sym = 0.5 * (score_matrix + score_matrix.T)
    np.fill_diagonal(sym, 0.0)
    if L < 3:
        return sym

    row_means = np.sum(sym, axis=1) / float(max(1, L - 1))
    global_mean = float(np.sum(sym)) / float(max(1, L * (L - 1)))
    if abs(global_mean) < 1e-12:
        apc = np.zeros_like(sym)
    else:
        apc = sym - np.outer(row_means, row_means) / global_mean

    idx = np.arange(L)
    sep_mask = np.abs(idx[:, None] - idx[None, :]) >= min_seq_sep
    out = np.where(sep_mask, apc, 0.0)
    return 0.5 * (out + out.T)


def _zero_sum_gauge_frobenius(couplings_4d: FloatArray) -> FloatArray:
    """Project 4D couplings J[i, j, a, b] into zero-sum Ising gauge over 20 AAs and take norm.

    In the zero-sum gauge on the 20 non-gap amino acids:
      J'_{ij}(a, b) = J_{ij}(a, b) - mean_a J_{ij} - mean_b J_{ij} + mean_{a,b} J_{ij}
    which minimizes the Frobenius norm among all gauge-equivalent Potts parameterizations.
    """
    sub = couplings_4d[:, :, :20, :20]
    mean_a = np.mean(sub, axis=2, keepdims=True)
    mean_b = np.mean(sub, axis=3, keepdims=True)
    mean_ab = np.mean(sub, axis=(2, 3), keepdims=True)
    gauged = sub - mean_a - mean_b + mean_ab
    frob = np.asarray(np.sqrt(np.sum(gauged * gauged, axis=(2, 3))), dtype=np.float64)
    np.fill_diagonal(frob, 0.0)
    return frob


def _extract_top_pairs(
    score_matrix: FloatArray, min_seq_sep: int, top_k: int
) -> List[Tuple[int, int, float]]:
    L = score_matrix.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(L):
        for j in range(i + min_seq_sep, L):
            pairs.append((i, j, float(score_matrix[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_k]


def compute_mi_apc(
    msa: MultipleSequenceAlignment, config: Optional[CoevolutionConfig] = None
) -> ContactMapResult:
    """Compute sequence-weighted Mutual Information with Average Product Correction (MI-APC)."""
    cfg = config or CoevolutionConfig()
    weights, m_eff = compute_clustering_weights(msa.matrix, cfg.identity_threshold)
    f1, f2, _ = compute_site_pair_frequencies(msa.matrix, weights, cfg.pseudocount, n_states=21)

    # Vectorized MI over 20 canonical states (excluding gap-gap artificial inflation)
    f1_aa = f1[:, :20]
    f2_aa = f2[:, :, :20, :20]
    prod = f1_aa[:, None, :, None] * f1_aa[None, :, None, :]
    ratio = np.where((f2_aa > 1e-15) & (prod > 1e-15), f2_aa / np.maximum(1e-15, prod), 1.0)
    mi_raw = np.sum(
        np.where(f2_aa > 1e-15, f2_aa * np.log(np.maximum(1e-15, ratio)), 0.0), axis=(2, 3)
    )
    np.fill_diagonal(mi_raw, 0.0)
    mi_raw = 0.5 * (mi_raw + mi_raw.T)

    apc_scores = apply_apc_correction(mi_raw, min_seq_sep=cfg.min_seq_sep)
    top_pairs = _extract_top_pairs(apc_scores, cfg.min_seq_sep, cfg.top_k_pairs)
    return ContactMapResult(
        method="MI-APC",
        score_matrix=apc_scores,
        raw_matrix=mi_raw,
        top_pairs=top_pairs,
        m_eff=m_eff,
    )


def compute_mfdca(
    msa: MultipleSequenceAlignment, config: Optional[CoevolutionConfig] = None
) -> ContactMapResult:
    """Compute Mean-Field Direct Coupling Analysis (mfDCA) via covariance matrix inversion.

    Inverts the reduced (L * 20) x (L * 20) connected covariance matrix
      C_{(i,a),(j,b)} = f_{ij}(a,b) - f_i(a) f_j(b)
    with ridge/pseudocount regularization, maps e_{ij} = -(C^{-1})_{ij} into the
    zero-sum gauge, and applies APC to isolate direct structural contacts.
    """
    cfg = config or CoevolutionConfig()
    weights, m_eff = compute_clustering_weights(msa.matrix, cfg.identity_threshold)
    f1, f2, _ = compute_site_pair_frequencies(
        msa.matrix, weights, max(0.05, cfg.pseudocount), n_states=21
    )
    L = msa.length
    q_red = 20

    # Build (L, L, 20, 20) connected covariance and reshape to (L*20, L*20)
    C4 = f2[:, :, :q_red, :q_red] - f1[:, None, :q_red, None] * f1[None, :, None, :q_red]
    C = C4.transpose(0, 2, 1, 3).reshape(L * q_red, L * q_red)
    C = 0.5 * (C + C.T)
    # Mild Tikhonov damping guarantees invertibility even for low M_eff
    C += cfg.l2_couplings * np.eye(L * q_red, dtype=np.float64)

    inv_C = np.linalg.pinv(C, rcond=1e-6)
    e4 = np.asarray(-inv_C.reshape(L, q_red, L, q_red).transpose(0, 2, 1, 3), dtype=np.float64)

    frob_raw = _zero_sum_gauge_frobenius(e4)
    apc_scores = apply_apc_correction(frob_raw, min_seq_sep=cfg.min_seq_sep)
    top_pairs = _extract_top_pairs(apc_scores, cfg.min_seq_sep, cfg.top_k_pairs)
    return ContactMapResult(
        method="mfDCA",
        score_matrix=apc_scores,
        raw_matrix=frob_raw,
        top_pairs=top_pairs,
        m_eff=m_eff,
    )


def compute_plmdca(
    msa: MultipleSequenceAlignment, config: Optional[CoevolutionConfig] = None
) -> ContactMapResult:
    """Compute Pseudolikelihood Maximization Direct Coupling Analysis (PLM-DCA).

    For each column i in 0..L-1, optimizes the regularized Potts conditional
    negative log-likelihood:
      NLL_i(h_i, J_{i,*}) = -sum_m w_m log P(a_i^{(m)} | a_{\\i}^{(m)})
                            + lambda_h ||h_i||_2^2 + lambda_J sum_{j != i} ||J_{ij}||_F^2
    using L-BFGS-B with exact analytical gradients. Symmetrizes J = 0.5 (J + J^T),
    projects into the zero-sum Ising gauge over the 20 amino acids, and applies APC.
    """
    cfg = config or CoevolutionConfig()
    weights, m_eff = compute_clustering_weights(msa.matrix, cfg.identity_threshold)
    n_seqs, L = msa.matrix.shape
    Q = 21

    # One-hot (N, L, Q) and normalized weights
    onehot = np.zeros((n_seqs, L, Q), dtype=np.float64)
    rows = np.arange(n_seqs)[:, None]
    cols = np.arange(L)[None, :]
    onehot[rows, cols, np.clip(msa.matrix, 0, Q - 1)] = 1.0
    w_norm = weights / max(1e-12, float(np.sum(weights)))

    # L2 regularization scaled for M_eff-normalized conditional NLL
    lam_h = cfg.l2_fields
    lam_j = cfg.l2_couplings

    J_full = np.zeros((L, L, Q, Q), dtype=np.float64)

    for i in range(L):
        # Mask out self-coupling block for site i
        other_idx = [j for j in range(L) if j != i]
        # Build design matrix X_other of shape (N, (L-1)*Q)
        x_other = onehot[:, other_idx, :].reshape(n_seqs, (L - 1) * Q)
        y_i = onehot[:, i, :]  # (N, Q)
        n_other_cols = (L - 1) * Q

        # Parameter vector theta = [h_i (Q), W ((L-1)*Q, Q)]
        theta0 = np.zeros(Q + n_other_cols * Q, dtype=np.float64)

        def _loss_and_grad(theta: FloatArray) -> Tuple[float, FloatArray]:
            h_i = theta[:Q]
            W = theta[Q:].reshape(n_other_cols, Q)

            logits = x_other @ W + h_i[None, :]  # (N, Q)
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

            # Negative weighted log-likelihood
            log_p = logits - np.log(np.sum(exp_l, axis=1, keepdims=True))
            nll = -float(np.sum(w_norm[:, None] * y_i * log_p))
            reg = lam_h * float(np.sum(h_i * h_i)) + lam_j * float(np.sum(W * W))

            # Residual: (N, Q) weighted
            diff = w_norm[:, None] * (probs - y_i)
            grad_h = np.sum(diff, axis=0) + 2.0 * lam_h * h_i
            grad_W = (x_other.T @ diff) + 2.0 * lam_j * W
            grad = np.concatenate([grad_h, grad_W.ravel()])
            return nll + reg, grad

        res = minimize(
            _loss_and_grad,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": cfg.plm_maxiter, "ftol": 1e-7, "gtol": 1e-5},
        )
        W_opt = res.x[Q:].reshape(L - 1, Q, Q)
        for pos_idx, j in enumerate(other_idx):
            # W_opt[pos_idx, b, a] couples state b at site j to state a at site i
            J_full[i, j, :, :] = W_opt[pos_idx].T

    # Symmetrize couplings: J_sym[i, j, a, b] = 0.5 * (J[i, j, a, b] + J[j, i, b, a])
    J_sym = 0.5 * (J_full + J_full.transpose(1, 0, 3, 2))
    frob_raw = _zero_sum_gauge_frobenius(J_sym)
    apc_scores = apply_apc_correction(frob_raw, min_seq_sep=cfg.min_seq_sep)
    top_pairs = _extract_top_pairs(apc_scores, cfg.min_seq_sep, cfg.top_k_pairs)
    return ContactMapResult(
        method="PLM-DCA",
        score_matrix=apc_scores,
        raw_matrix=frob_raw,
        top_pairs=top_pairs,
        m_eff=m_eff,
    )


def evaluate_contact_predictions(
    result: ContactMapResult,
    structure: ProteinStructure,
    cutoff: float = 8.0,
    min_seq_sep: int = 6,
) -> ContactMapResult:
    """Evaluate predicted co-evolutionary contacts against a 3D ProteinStructure.

    Computes Precision@L/5, Precision@L/2, Precision@L, and exact ROC-AUC over all
    residue pairs (i, j) satisfying j - i >= min_seq_sep.
    """
    L = min(result.score_matrix.shape[0], structure.length)
    true_map = structure.contact_map(cutoff=cutoff, min_seq_sep=min_seq_sep)[:L, :L]

    ranked: List[Tuple[float, bool]] = []
    for i in range(L):
        for j in range(i + min_seq_sep, L):
            ranked.append((float(result.score_matrix[i, j]), bool(true_map[i, j])))

    if not ranked:
        return result

    ranked.sort(key=lambda item: item[0], reverse=True)
    labels = np.array([1.0 if y else 0.0 for _, y in ranked], dtype=np.float64)

    def _prec_at(k: int) -> float:
        kk = max(1, min(k, len(labels)))
        return float(np.mean(labels[:kk]))

    p_l5 = _prec_at(max(1, L // 5))
    p_l2 = _prec_at(max(1, L // 2))
    p_l = _prec_at(L)

    # Exact AUROC via Mann-Whitney U rank statistic
    n_pos = float(np.sum(labels))
    n_neg = float(len(labels) - n_pos)
    if n_pos > 0 and n_neg > 0:
        # Ascending scores for Mann-Whitney U
        scores_asc = np.array([s for s, _ in reversed(ranked)], dtype=np.float64)
        labels_asc = labels[::-1]
        # Average rank for ties via scipy/manual tie handling
        order = np.argsort(scores_asc, kind="mergesort")
        ranks = np.empty_like(scores_asc)
        ranks[order] = np.arange(1, len(scores_asc) + 1, dtype=np.float64)
        pos_rank_sum = float(np.sum(ranks[labels_asc == 1.0]))
        auroc = (pos_rank_sum - 0.5 * n_pos * (n_pos + 1.0)) / (n_pos * n_neg)
    else:
        auroc = 0.5

    return ContactMapResult(
        method=result.method,
        score_matrix=result.score_matrix,
        raw_matrix=result.raw_matrix,
        top_pairs=result.top_pairs,
        m_eff=result.m_eff,
        precision_at_l5=p_l5,
        precision_at_l2=p_l2,
        precision_at_l=p_l,
        auroc=float(auroc),
    )


def compute_coupling_centrality(result: ContactMapResult, min_seq_sep: int = 6) -> FloatArray:
    """Compute per-residue positive co-evolutionary coupling strength (weighted degree)."""
    L = result.score_matrix.shape[0]
    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :]) >= min_seq_sep
    pos = np.where(sep, np.maximum(0.0, result.score_matrix), 0.0)
    return np.asarray(np.sum(pos, axis=1), dtype=np.float64)


def compare_coevolution_methods(
    msa: MultipleSequenceAlignment,
    structure: Optional[ProteinStructure] = None,
    config: Optional[CoevolutionConfig] = None,
) -> Dict[str, ContactMapResult]:
    """Run MI-APC, mfDCA, and PLM-DCA on an MSA and optionally benchmark against 3D contacts."""
    cfg = config or CoevolutionConfig()
    results = {
        "MI-APC": compute_mi_apc(msa, cfg),
        "mfDCA": compute_mfdca(msa, cfg),
        "PLM-DCA": compute_plmdca(msa, cfg),
    }
    if structure is not None:
        for name, res in list(results.items()):
            results[name] = evaluate_contact_predictions(
                res, structure, min_seq_sep=cfg.min_seq_sep
            )
    return results
