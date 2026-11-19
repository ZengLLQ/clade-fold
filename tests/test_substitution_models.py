"""Tests for BLOSUM62, PAM250, and CTMC protein substitution models (JTT, WAG, LG)."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.substitution_models import (
    BLOSUM62_MATRIX,
    PAM250_MATRIX,
    SubstitutionModel,
    compute_model_statistics,
    discrete_gamma_rates,
    get_scoring_matrix,
    list_available_models,
    score_substitution_pair,
)


def test_blosum62_and_pam250_invariants() -> None:
    for mat in (BLOSUM62_MATRIX, PAM250_MATRIX):
        assert mat.shape == (20, 20)
        assert np.allclose(mat, mat.T)
        assert np.all(np.diag(mat) > 0)
    assert score_substitution_pair("W", "W", BLOSUM62_MATRIX) == 11.0
    assert score_substitution_pair("D", "E", get_scoring_matrix("BLOSUM62")) == 2.0
    assert score_substitution_pair("A", "-", BLOSUM62_MATRIX) < 0.0
    with pytest.raises(ValueError):
        get_scoring_matrix("UNKNOWN")


@pytest.mark.parametrize("model_name", list_available_models())
def test_ctmc_conservation_and_semigroup_laws(model_name: str) -> None:
    model = SubstitutionModel.from_name(model_name)
    stats = compute_model_statistics(model)
    assert stats["q_row_sum_max_err"] < 1e-12
    assert stats["p_row_sum_max_err"] < 1e-12
    assert stats["detailed_balance_max_err"] < 1e-12
    assert stats["stationary_max_err"] < 1e-11
    assert stats["mean_substitution_rate"] == pytest.approx(1.0, abs=1e-10)

    # P(0) = I and Chapman-Kolmogorov P(t1 + t2) = P(t1) @ P(t2)
    assert np.allclose(model.transition_matrix(0.0), np.eye(20), atol=1e-12)
    p1 = model.transition_matrix(0.15)
    p2 = model.transition_matrix(0.35)
    p12 = model.transition_matrix(0.50)
    assert np.allclose(p1 @ p2, p12, atol=1e-10)

    # Batched rates match individual matrix exponentials
    rates = np.array([0.25, 1.0, 2.5], dtype=np.float64)
    batched = model.batched_transition_matrices(0.4, rates)
    for k, r in enumerate(rates):
        assert np.allclose(batched[k], model.transition_matrix(0.4 * float(r)), atol=1e-10)


def test_transition_derivative_matches_finite_difference() -> None:
    model = SubstitutionModel.from_name("WAG")
    t = 0.25
    eps = 1e-6
    fd = (model.transition_matrix(t + eps) - model.transition_matrix(t - eps)) / (2.0 * eps)
    ana = model.transition_derivative(t)
    assert np.allclose(ana, fd, atol=1e-5)


def test_discrete_gamma_rates_mean_and_monotonicity() -> None:
    for alpha in (0.3, 1.0, 5.0):
        rates = discrete_gamma_rates(alpha, n_categories=4)
        assert rates.shape == (4,)
        assert float(np.mean(rates)) == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(rates) > 0.0)
    assert np.allclose(discrete_gamma_rates(1.0, n_categories=1), np.array([1.0]))
