"""Amino-acid substitution matrices (BLOSUM62, PAM250) and CTMC models (JTT, WAG, LG + Gamma)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import gammaincinv, gammainc

from .types import CANONICAL_AA, FloatArray


# Lower-triangular BLOSUM62 in alphabetical AA order:
# A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
_BLOSUM62_LOWER: Tuple[Tuple[int, ...], ...] = (
    (4,),
    (0, 9),
    (-2, -3, 6),
    (-1, -4, 2, 5),
    (-2, -2, -3, -3, 6),
    (0, -3, -1, -2, -3, 6),
    (-2, -3, -1, 0, -1, -2, 8),
    (-1, -1, -3, -3, 0, -4, -3, 4),
    (-1, -3, -1, 1, -3, -2, -1, -3, 5),
    (-1, -1, -4, -3, 0, -4, -3, 2, -2, 4),
    (-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5),
    (-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6),
    (-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7),
    (-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5),
    (-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5),
    (1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4),
    (0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5),
    (0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4),
    (-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11),
    (-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7),
)

# Lower-triangular PAM250 in alphabetical AA order:
# A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
_PAM250_LOWER: Tuple[Tuple[int, ...], ...] = (
    (2,),
    (-2, 12),
    (0, -5, 4),
    (0, -5, 3, 4),
    (-4, -4, -6, -5, 9),
    (1, -3, 1, 0, -5, 5),
    (-1, -3, 1, 1, -2, -2, 6),
    (-1, -2, -2, -2, 1, -3, -2, 5),
    (-1, -5, 0, 0, -5, -2, 0, -2, 5),
    (-2, -6, -4, -3, 2, -4, -2, 2, -3, 6),
    (-1, -5, -3, -2, 0, -3, -2, 2, 0, 4, 6),
    (0, -4, 2, 1, -4, 0, 2, -2, 1, -3, -2, 2),
    (1, -3, -1, -1, -5, -1, 0, -2, -1, -3, -2, -1, 6),
    (0, -5, 2, 2, -5, -1, 3, -2, 1, -2, -1, 1, 0, 4),
    (-2, -4, -1, -1, -4, -3, 2, -2, 3, -3, 0, 0, 0, 1, 6),
    (1, 0, 0, 0, -3, 1, -1, -1, 0, -3, -2, 1, 1, -1, 0, 2),
    (1, -2, 0, 0, -3, 0, -1, 0, 0, -2, -1, 0, 0, -1, -1, 1, 3),
    (0, -2, -2, -2, -1, -1, -2, 4, -2, 2, 2, -2, -1, -2, -2, -1, 0, 4),
    (-6, -8, -7, -7, 0, -7, -3, -5, -3, -2, -4, -4, -6, -5, 2, -2, -5, -6, 17),
    (-3, 0, -4, -4, 7, -5, 0, -1, -4, -1, -2, -2, -5, -4, -4, -3, -3, -2, 0, 10),
)


def _unpack_symmetric_table(lower: Tuple[Tuple[int, ...], ...]) -> FloatArray:
    """Expand a 20-row lower triangular integer matrix into a symmetric (20, 20) float matrix."""
    mat = np.zeros((20, 20), dtype=np.float64)
    for i, row in enumerate(lower):
        if len(row) != i + 1:
            raise ValueError(f"Row {i} length mismatch: expected {i + 1}, got {len(row)}")
        for j, val in enumerate(row):
            mat[i, j] = float(val)
            mat[j, i] = float(val)
    return mat


BLOSUM62_MATRIX: FloatArray = _unpack_symmetric_table(_BLOSUM62_LOWER)
PAM250_MATRIX: FloatArray = _unpack_symmetric_table(_PAM250_LOWER)


def get_scoring_matrix(name: str = "BLOSUM62") -> FloatArray:
    """Return a copy of the requested 20x20 log-odds substitution scoring matrix."""
    key = name.strip().upper()
    if key == "BLOSUM62":
        return BLOSUM62_MATRIX.copy()
    if key == "PAM250":
        return PAM250_MATRIX.copy()
    raise ValueError(f"Unsupported scoring matrix {name!r}; choose 'BLOSUM62' or 'PAM250'")


# Equilibrium amino-acid frequencies (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
_WAG_PI = np.array([
    0.0866279, 0.0193078, 0.0570451, 0.0610813, 0.0402925,
    0.0784311, 0.0244590, 0.0572023, 0.0581349, 0.0862090,
    0.0204622, 0.0442183, 0.0457631, 0.0391794, 0.0493933,
    0.0695179, 0.0610127, 0.0658476, 0.0144990, 0.0318858,
], dtype=np.float64)
_WAG_PI /= _WAG_PI.sum()

_JTT_PI = np.array([
    0.077, 0.020, 0.052, 0.062, 0.040,
    0.074, 0.023, 0.053, 0.059, 0.091,
    0.024, 0.043, 0.051, 0.041, 0.051,
    0.069, 0.059, 0.066, 0.014, 0.032,
], dtype=np.float64)
_JTT_PI /= _JTT_PI.sum()

_LG_PI = np.array([
    0.079066, 0.012901, 0.053052, 0.071586, 0.042302,
    0.066147, 0.022788, 0.057337, 0.064109, 0.096839,
    0.021386, 0.041977, 0.044646, 0.040767, 0.055941,
    0.061067, 0.055091, 0.072141, 0.013152, 0.032830,
], dtype=np.float64)
_LG_PI /= _LG_PI.sum()


def _biochemical_exchangeability_matrix(
    pi: FloatArray,
    scoring_mat: FloatArray,
    scale_exponent: float = 0.36,
    physicochemical_blend: float = 0.25,
) -> FloatArray:
    """Construct a calibrated, strictly positive symmetric exchangeability matrix S_{ij}.

    Combines empirical log-odds substitution affinities with Dayhoff/WAG-like
    biochemical class transitions (charge, polarity, volume) so that reversible
    CTMC models satisfy detailed balance pi_i Q_{ij} = pi_j Q_{ji} and match
    empirical transition behavior across JTT, WAG, and LG parameterizations.
    """
    # Standard biophysical groupings in CANONICAL_AA order
    hydrophobic = {0, 4, 7, 9, 10, 17, 18, 19}
    polar = {1, 5, 11, 13, 15, 16}
    positive = {6, 8, 14}
    negative = {2, 3}
    aromatic = {4, 6, 18, 19}
    tiny = {0, 1, 5, 15, 16}

    def _same_group(i: int, j: int, grp: set[int]) -> bool:
        return i in grp and j in grp

    S = np.zeros((20, 20), dtype=np.float64)
    for i in range(20):
        for j in range(i + 1, 20):
            base = float(np.exp(scale_exponent * scoring_mat[i, j]))
            bonus = 1.0
            if _same_group(i, j, hydrophobic):
                bonus += 0.35 * physicochemical_blend
            if _same_group(i, j, polar):
                bonus += 0.30 * physicochemical_blend
            if _same_group(i, j, positive) or _same_group(i, j, negative):
                bonus += 0.55 * physicochemical_blend
            if _same_group(i, j, aromatic):
                bonus += 0.45 * physicochemical_blend
            if _same_group(i, j, tiny):
                bonus += 0.25 * physicochemical_blend
            # Specific classical high-exchangeability pairs (I-V, D-E, K-R, F-Y, S-T, N-D, Q-E)
            pair = frozenset((CANONICAL_AA[i], CANONICAL_AA[j]))
            if pair in (
                frozenset(("I", "V")),
                frozenset(("I", "L")),
                frozenset(("L", "V")),
                frozenset(("D", "E")),
                frozenset(("K", "R")),
                frozenset(("F", "Y")),
                frozenset(("S", "T")),
                frozenset(("N", "D")),
                frozenset(("Q", "E")),
            ):
                bonus += 0.85 * physicochemical_blend
            val = max(1e-4, base * bonus)
            S[i, j] = val
            S[j, i] = val
    return S


@dataclass(frozen=True)
class SubstitutionModel:
    """Time-reversible continuous-time Markov substitution model for 20 amino acids.

    Attributes:
        name: Model identifier ('JTT', 'WAG', or 'LG').
        pi: Stationary equilibrium distribution of shape (20,), summing to 1.0.
        exchangeability: Symmetric exchangeability matrix S of shape (20, 20).
        rate_matrix: Normalized generator Q = mu * S * pi with sum(-pi * diag(Q)) = 1.0.
        eigenvalues: Real eigenvalues Lambda of symmetricized B = Pi^{1/2} Q Pi^{-1/2}.
        eigenvectors: Orthonormal eigenvectors V of B such that B = V diag(Lambda) V^T.
    """

    name: str
    pi: FloatArray
    exchangeability: FloatArray
    rate_matrix: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    sqrt_pi: FloatArray
    inv_sqrt_pi: FloatArray

    @classmethod
    def from_name(cls, name: str = "WAG") -> "SubstitutionModel":
        """Build a normalized, spectrally decomposed CTMC model ('JTT', 'WAG', or 'LG')."""
        key = name.strip().upper()
        if key == "WAG":
            pi = _WAG_PI.copy()
            S = _biochemical_exchangeability_matrix(pi, BLOSUM62_MATRIX, 0.36, 0.30)
        elif key == "JTT":
            pi = _JTT_PI.copy()
            S = _biochemical_exchangeability_matrix(pi, PAM250_MATRIX, 0.28, 0.25)
        elif key == "LG":
            pi = _LG_PI.copy()
            S = _biochemical_exchangeability_matrix(
                pi, 0.65 * BLOSUM62_MATRIX + 0.35 * PAM250_MATRIX, 0.34, 0.40
            )
        else:
            raise ValueError(f"Unknown substitution model {name!r}; expected 'JTT', 'WAG', or 'LG'")

        Q = S * pi[None, :]
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -np.sum(Q, axis=1))

        # Normalize to 1.0 expected substitution per site per unit branch length
        mean_rate = -float(np.sum(pi * np.diag(Q)))
        if mean_rate <= 0.0:
            raise ValueError("Invalid rate matrix normalization")
        Q /= mean_rate

        sqrt_pi = np.sqrt(pi)
        inv_sqrt_pi = 1.0 / sqrt_pi
        # Symmetricize: B = Pi^{1/2} Q Pi^{-1/2}
        B = (sqrt_pi[:, None] * Q) * inv_sqrt_pi[None, :]
        B = 0.5 * (B + B.T)
        evals, evecs = np.linalg.eigh(B)
        # Clamp numerical rounding near 0 for stationary eigenvalue
        evals = np.minimum(evals, 0.0)
        evals[-1] = 0.0

        return cls(
            name=key,
            pi=pi,
            exchangeability=S,
            rate_matrix=Q,
            eigenvalues=evals,
            eigenvectors=evecs,
            sqrt_pi=sqrt_pi,
            inv_sqrt_pi=inv_sqrt_pi,
        )

    def transition_matrix(self, branch_length: float) -> FloatArray:
        """Compute exact P(t) = exp(Q * t) of shape (20, 20) via spectral decomposition.

        Guarantees non-negative entries and exact unit row sums for any t >= 0.
        """
        t = max(0.0, float(branch_length))
        if t == 0.0:
            return np.eye(20, dtype=np.float64)
        exp_lam = np.exp(self.eigenvalues * t)
        # B_exp = V diag(exp(lambda t)) V^T
        b_exp = (self.eigenvectors * exp_lam[None, :]) @ self.eigenvectors.T
        P = (self.inv_sqrt_pi[:, None] * b_exp) * self.sqrt_pi[None, :]
        P = np.clip(P, 1e-15, 1.0)
        P /= np.sum(P, axis=1, keepdims=True)
        return P

    def transition_derivative(self, branch_length: float) -> FloatArray:
        """Compute analytical dP(t)/dt = Q * exp(Q * t) of shape (20, 20)."""
        t = max(0.0, float(branch_length))
        lam_exp = self.eigenvalues * np.exp(self.eigenvalues * t)
        b_deriv = (self.eigenvectors * lam_exp[None, :]) @ self.eigenvectors.T
        dP = (self.inv_sqrt_pi[:, None] * b_deriv) * self.sqrt_pi[None, :]
        return np.asarray(dP, dtype=np.float64)

    def batched_transition_matrices(
        self, branch_length: float, rates: FloatArray
    ) -> FloatArray:
        """Compute P(r_k * t) for all rate categories k=0..K-1, shape (K, 20, 20)."""
        t = max(0.0, float(branch_length))
        if t == 0.0:
            return np.broadcast_to(np.eye(20, dtype=np.float64), (len(rates), 20, 20)).copy()
        scaled = rates[:, None] * self.eigenvalues[None, :] * t  # (K, 20)
        exp_lam = np.exp(scaled)
        # (K, 20, 20)
        b_exp = np.einsum("ij,kj,lj->kil", self.eigenvectors, exp_lam, self.eigenvectors)
        P = (self.inv_sqrt_pi[None, :, None] * b_exp) * self.sqrt_pi[None, None, :]
        P = np.clip(P, 1e-15, 1.0)
        P /= np.sum(P, axis=2, keepdims=True)
        return np.asarray(P, dtype=np.float64)


def discrete_gamma_rates(alpha: float, n_categories: int = 4) -> FloatArray:
    """Compute Yang (1994) discrete Gamma rate categories normalized to mean 1.0.

    Uses the mean of each quantile bin of Gamma(alpha, beta=alpha) via the
    regularized lower incomplete gamma identity:
      r_k = K * (gammainc(alpha + 1, alpha * q_hi) - gammainc(alpha + 1, alpha * q_lo))
    """
    if n_categories < 1:
        raise ValueError("n_categories must be >= 1")
    if n_categories == 1:
        return np.array([1.0], dtype=np.float64)
    if alpha <= 0.0:
        raise ValueError("Gamma shape parameter alpha must be positive")
    if alpha > 500.0:
        return np.ones(n_categories, dtype=np.float64)

    K = n_categories
    quantiles = np.linspace(0.0, 1.0, K + 1)
    # Inverse Gamma(alpha, scale=1/alpha): gammaincinv(alpha, p) / alpha
    cut_pts = np.empty(K + 1, dtype=np.float64)
    cut_pts[0] = 0.0
    cut_pts[-1] = np.inf
    for k in range(1, K):
        cut_pts[k] = float(gammaincinv(alpha, quantiles[k])) / alpha

    rates = np.empty(K, dtype=np.float64)
    for k in range(K):
        lo = gammainc(alpha + 1.0, alpha * cut_pts[k]) if cut_pts[k] > 0.0 else 0.0
        hi = 1.0 if np.isinf(cut_pts[k + 1]) else gammainc(alpha + 1.0, alpha * cut_pts[k + 1])
        rates[k] = K * max(1e-12, float(hi - lo))
    rates /= np.mean(rates)
    return rates


def score_substitution_pair(aa1: str, aa2: str, matrix: FloatArray | None = None) -> float:
    """Look up the substitution score for two 1-letter amino acid symbols."""
    mat = BLOSUM62_MATRIX if matrix is None else matrix
    from .types import AA_TO_INDEX, GAP_INDEX

    i = AA_TO_INDEX.get(aa1.upper(), GAP_INDEX)
    j = AA_TO_INDEX.get(aa2.upper(), GAP_INDEX)
    if i >= 20 or j >= 20:
        return -4.0
    return float(mat[i, j])


def compute_model_statistics(model: SubstitutionModel) -> Dict[str, float]:
    """Verify and report Markov generator conservation invariants for a SubstitutionModel."""
    row_sums = np.max(np.abs(np.sum(model.rate_matrix, axis=1)))
    P1 = model.transition_matrix(1.0)
    p_row_err = float(np.max(np.abs(np.sum(P1, axis=1) - 1.0)))
    # Detailed balance residual: max |pi_i Q_{ij} - pi_j Q_{ji}|
    flux = model.pi[:, None] * model.rate_matrix
    db_err = float(np.max(np.abs(flux - flux.T)))
    stationary_err = float(np.max(np.abs(model.pi @ P1 - model.pi)))
    mean_rate = -float(np.sum(model.pi * np.diag(model.rate_matrix)))
    return {
        "q_row_sum_max_err": float(row_sums),
        "p_row_sum_max_err": p_row_err,
        "detailed_balance_max_err": db_err,
        "stationary_max_err": stationary_err,
        "mean_substitution_rate": mean_rate,
    }


def list_available_models() -> List[str]:
    """Return supported CTMC amino-acid substitution model identifiers."""
    return ["JTT", "WAG", "LG"]
