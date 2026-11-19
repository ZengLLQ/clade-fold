"""Example 01: Progressive Gotoh MSA, conservation scoring, and DCA contact inference."""

from __future__ import annotations

from clade_fold.ascii_viz import render_contact_map_ascii, render_msa_alignment_ascii
from clade_fold.coevolution import CoevolutionConfig, compare_coevolution_methods
from clade_fold.msa_engine import (
    AlignmentConfig,
    build_progressive_msa,
    compute_conservation_profile,
    summarise_msa,
)
from clade_fold.synthetic_benchmark import generate_coupled_benchmark


def main() -> None:
    suite = generate_coupled_benchmark(length=32, n_taxa=8, n_coev_seqs=140, seed=42)

    print("=== Step 1: Progressive + Iterative Gotoh MSA ===")
    msa = build_progressive_msa(
        suite.unaligned_sequences,
        AlignmentConfig(matrix_name="BLOSUM62", guide_method="nj", refinement_rounds=2),
    )
    cons = compute_conservation_profile(msa)
    print(render_msa_alignment_ascii(msa, conservation=cons, max_seqs=8))
    print("MSA Metrics:", summarise_msa(msa))

    print("\n=== Step 2: Evolutionary Co-Variation Benchmark (MI-APC vs mfDCA vs PLM-DCA) ===")
    cfg = CoevolutionConfig(min_seq_sep=6, plm_maxiter=45, top_k_pairs=30)
    results = compare_coevolution_methods(suite.coev_msa, structure=suite.structure, config=cfg)

    for method, res in results.items():
        print(
            f"{method:<8s} | M_eff={res.m_eff:6.1f} | "
            f"P@L/5={res.precision_at_l5 or 0.0:.3f} | "
            f"P@L={res.precision_at_l or 0.0:.3f} | "
            f"AUROC={res.auroc or 0.0:.3f}"
        )

    plm = results["PLM-DCA"]
    true_cmap = suite.structure.contact_map(cutoff=8.0, min_seq_sep=6)
    print("\n" + render_contact_map_ascii(plm.score_matrix, true_contacts=true_cmap, top_k=24))


if __name__ == "__main__":
    main()
