"""Example 03: Felsenstein pruning ML, NNI topology search, Bayesian MCMC, and ASR."""

from __future__ import annotations

from clade_fold.ascii_viz import render_tree_ascii
from clade_fold.io_formats import write_newick
from clade_fold.phylogenetics import (
    PhyloConfig,
    compute_aic_bic,
    infer_phylogeny_pipeline,
    robinson_foulds_distance,
)
from clade_fold.pipeline import build_kimura_nj_tree
from clade_fold.substitution_models import SubstitutionModel, compute_model_statistics
from clade_fold.synthetic_benchmark import generate_coupled_benchmark


def main() -> None:
    for m_name in ("JTT", "WAG", "LG"):
        stats = compute_model_statistics(SubstitutionModel.from_name(m_name))
        print(
            f"Model {m_name}: DB_err={stats['detailed_balance_max_err']:.2e}, "
            f"P_row_err={stats['p_row_sum_max_err']:.2e}"
        )

    suite = generate_coupled_benchmark(
        length=48, n_taxa=8, n_coev_seqs=60, model_name="WAG", seed=42
    )
    seed_tree = build_kimura_nj_tree(suite.phylo_msa)

    cfg = PhyloConfig(model_name="WAG", n_gamma_categories=4, mcmc_steps=120, seed=42)
    ml_tree, opt_alpha, ml_ll, mcmc_res, asr = infer_phylogeny_pipeline(
        suite.phylo_msa, seed_tree, cfg, run_mcmc=True
    )

    rf_dist, norm_rf = robinson_foulds_distance(suite.true_tree, ml_tree)
    ic = compute_aic_bic(ml_ll, ml_tree, suite.phylo_msa.length)

    print(f"\nML Log-Likelihood : {ml_ll:.3f} (alpha={opt_alpha:.3f})")
    print(f"AIC / BIC         : {ic['aic']:.2f} / {ic['bic']:.2f}")
    print(f"RF vs True Tree   : {rf_dist} (normalized={norm_rf:.3f})")
    print(f"ML Newick         : {write_newick(ml_tree, include_support=False)}")
    if mcmc_res is not None:
        print(
            f"MCMC Acceptance   : {mcmc_res.acceptance_rate:.3f} | "
            f"ESS={mcmc_res.ess_log_likelihood:.1f} | "
            f"Mean TL={mcmc_res.mean_tree_length:.3f}"
        )
        print("\n" + render_tree_ascii(mcmc_res.map_tree))
    print(f"\nASR Mean Posterior Confidence: {asr.mean_posterior_confidence:.3f}")
    print(f"Root Marginal Seq : {asr.marginal_sequences.get('Node_0', '')}")
    print(f"Root Joint Seq    : {asr.joint_sequences.get('Node_0', '')}")


if __name__ == "__main__":
    main()
