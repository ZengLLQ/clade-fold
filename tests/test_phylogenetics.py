"""Tests for Felsenstein pruning, ML branch/NNI search, Bayesian MCMC, and Marginal/Joint ASR."""

from __future__ import annotations

import numpy as np
import pytest

from clade_fold.phylogenetics import (
    PhyloConfig,
    compute_aic_bic,
    felsenstein_log_likelihood,
    infer_phylogeny_pipeline,
    optimize_branch_lengths_ml,
    reconstruct_ancestral_sequences,
    robinson_foulds_distance,
    run_bayesian_mcmc,
    search_topology_nni,
)
from clade_fold.substitution_models import SubstitutionModel
from clade_fold.synthetic_benchmark import generate_random_phylo_tree, simulate_tree_evolution
from clade_fold.types import AA_TO_INDEX, MultipleSequenceAlignment, TreeNode


def test_felsenstein_pruning_matches_bruteforce_exact_sum() -> None:
    # 3-taxon tree ((A, B), C) with 1 internal root + 1 internal child = 2 internal nodes
    model = SubstitutionModel.from_name("WAG")
    tree = TreeNode(
        children=[
            TreeNode(
                children=[
                    TreeNode(name="A", branch_length=0.15),
                    TreeNode(name="B", branch_length=0.25),
                ],
                branch_length=0.10,
            ),
            TreeNode(name="C", branch_length=0.30),
        ]
    )
    msa = MultipleSequenceAlignment(
        names=["A", "B", "C"],
        sequences=["AD", "AE", "VD"],
    )
    ll_pruned = felsenstein_log_likelihood(tree, msa, model, alpha=1.0, n_gamma=1)

    # Brute-force sum over all 20 x 20 internal state pairs per site
    P_ab_parent = model.transition_matrix(0.10)
    P_a = model.transition_matrix(0.15)
    P_b = model.transition_matrix(0.25)
    P_c = model.transition_matrix(0.30)

    ll_brute = 0.0
    for s in range(2):
        idx_a = AA_TO_INDEX[msa.sequences[0][s]]
        idx_b = AA_TO_INDEX[msa.sequences[1][s]]
        idx_c = AA_TO_INDEX[msa.sequences[2][s]]
        site_prob = 0.0
        for r_state in range(20):
            for u_state in range(20):
                site_prob += (
                    float(model.pi[r_state])
                    * float(P_ab_parent[r_state, u_state])
                    * float(P_a[u_state, idx_a])
                    * float(P_b[u_state, idx_b])
                    * float(P_c[r_state, idx_c])
                )
        ll_brute += float(np.log(site_prob))

    assert ll_pruned == pytest.approx(ll_brute, abs=1e-10)


def test_ml_optimization_nni_and_asr_recovery() -> None:
    true_tree = generate_random_phylo_tree(n_taxa=6, seed=11, min_branch=0.08, max_branch=0.22)
    msa, true_anc, _ = simulate_tree_evolution(
        true_tree, length=90, model_name="WAG", alpha=1.2, seed=12
    )
    model = SubstitutionModel.from_name("WAG")

    # Perturb branch lengths and verify ML optimization strictly improves likelihood
    perturbed = true_tree.clone()
    for node in perturbed.postorder()[:-1]:
        node.branch_length = 0.8
    ll_bad = felsenstein_log_likelihood(perturbed, msa, model, alpha=1.0, n_gamma=4)
    opt_tree, opt_alpha, ll_opt = optimize_branch_lengths_ml(
        perturbed, msa, model, alpha=1.0, n_gamma=4, n_rounds=2, optimize_alpha=True
    )
    assert ll_opt > ll_bad

    nni_tree, ll_nni = search_topology_nni(opt_tree, msa, model, alpha=opt_alpha, n_gamma=4)
    assert ll_nni >= ll_opt - 1e-6
    ic = compute_aic_bic(ll_nni, nni_tree, msa.length)
    assert ic["aic"] > 0.0 and ic["bic"] > 0.0

    # Ancestral sequence reconstruction accuracy vs true root sequence
    asr = reconstruct_ancestral_sequences(true_tree, msa, model)
    assert asr.mean_posterior_confidence > 0.55
    true_root = true_anc["Node_0"]
    marg_root = asr.marginal_sequences["Node_0"]
    joint_root = asr.joint_sequences["Node_0"]
    marg_acc = sum(a == b for a, b in zip(true_root, marg_root)) / len(true_root)
    joint_acc = sum(a == b for a, b in zip(true_root, joint_root)) / len(true_root)
    assert marg_acc >= 0.70
    assert joint_acc >= 0.70


def test_bayesian_mcmc_and_robinson_foulds() -> None:
    tree = generate_random_phylo_tree(n_taxa=6, seed=31)
    msa, _, _ = simulate_tree_evolution(tree, length=50, model_name="LG", alpha=1.0, seed=32)
    cfg = PhyloConfig(model_name="LG", mcmc_steps=90, burnin_fraction=0.2, seed=33)
    ml_tree, _, _, mcmc, _ = infer_phylogeny_pipeline(msa, tree, cfg, run_mcmc=True)
    assert mcmc is not None
    assert len(mcmc.log_posterior_trace) == 90
    assert 0.0 < mcmc.acceptance_rate <= 1.0
    assert mcmc.ess_log_likelihood >= 1.0
    assert mcmc.mean_tree_length > 0.0
    assert all(0.0 <= p <= 1.0 for p in mcmc.clade_posteriors.values())

    rf_self, nrf_self = robinson_foulds_distance(ml_tree, ml_tree)
    assert rf_self == 0 and nrf_self == 0.0

    # Also test run_bayesian_mcmc directly
    model = SubstitutionModel.from_name("LG")
    direct = run_bayesian_mcmc(ml_tree, msa, model, cfg)
    assert direct.map_log_posterior >= -1e9
