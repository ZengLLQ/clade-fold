"""Command-line interface for CladeFold structural proteomics and phylogenetic analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional
import numpy as np

from .ascii_viz import (
    render_contact_map_ascii,
    render_dynamics_profile_ascii,
    render_msa_alignment_ascii,
    render_ramachandran_ascii,
    render_tree_ascii,
)
from .coevolution import (
    CoevolutionConfig,
    compute_mfdca,
    compute_mi_apc,
    compute_plmdca,
    evaluate_contact_predictions,
)
from .contact_folding import FoldingConfig, reconstruct_ca_fold
from .elastic_dynamics import ElasticNetworkConfig, run_elastic_network_analysis
from .io_formats import (
    parse_a3m,
    parse_fasta,
    parse_newick,
    parse_pdb,
    read_text_file,
    write_fasta,
    write_json_report,
    write_newick,
)
from .msa_engine import (
    AlignmentConfig,
    build_progressive_msa,
    compute_conservation_profile,
    summarise_msa,
)
from .phylogenetics import PhyloConfig, infer_phylogeny_pipeline, robinson_foulds_distance
from .pipeline import CladeFoldPipeline, build_kimura_nj_tree
from .structural_geometry import (
    analyze_structure,
    compute_tm_score,
    kabsch_superpose,
    summarize_dssp,
)
from .synthetic_benchmark import generate_coupled_benchmark


def _cmd_align(args: argparse.Namespace) -> int:
    records = parse_fasta(read_text_file(args.input))
    cfg = AlignmentConfig(
        matrix_name=args.matrix,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
        guide_method=args.guide,
        refinement_rounds=args.refine,
    )
    msa = build_progressive_msa(records, cfg)
    cons = compute_conservation_profile(msa)
    stats = summarise_msa(msa)

    if args.output:
        Path(args.output).write_text(
            write_fasta(list(zip(msa.names, msa.sequences))), encoding="utf-8"
        )
    print(render_msa_alignment_ascii(msa, conservation=cons))
    print("Alignment Summary:", json.dumps(stats, indent=2))
    return 0


def _cmd_coevolution(args: argparse.Namespace) -> int:
    msa = parse_a3m(read_text_file(args.msa))
    cfg = CoevolutionConfig(
        identity_threshold=args.id_thresh,
        pseudocount=args.pseudocount,
        min_seq_sep=args.min_sep,
        plm_maxiter=args.maxiter,
        top_k_pairs=args.top_k,
    )
    method = args.method.lower()
    if method == "mi":
        res = compute_mi_apc(msa, cfg)
    elif method == "mfdca":
        res = compute_mfdca(msa, cfg)
    else:
        res = compute_plmdca(msa, cfg)

    true_cmap = None
    if args.pdb:
        struct = parse_pdb(read_text_file(args.pdb))
        res = evaluate_contact_predictions(res, struct, min_seq_sep=cfg.min_seq_sep)
        true_cmap = struct.contact_map(cutoff=8.0, min_seq_sep=cfg.min_seq_sep)

    print(
        render_contact_map_ascii(
            res.score_matrix,
            true_contacts=true_cmap,
            top_k=args.top_k,
            min_seq_sep=cfg.min_seq_sep,
        )
    )
    summary = {
        "method": res.method,
        "m_eff": res.m_eff,
        "precision_at_l5": res.precision_at_l5,
        "precision_at_l2": res.precision_at_l2,
        "precision_at_l": res.precision_at_l,
        "auroc": res.auroc,
        "top_10_pairs": [
            {"i": i + 1, "j": j + 1, "score": round(s, 4)} for i, j, s in res.top_pairs[:10]
        ],
    }
    print("Coevolution Telemetry:", json.dumps(summary, indent=2))
    if args.json_out:
        write_json_report(summary, args.json_out)
    return 0


def _cmd_structure(args: argparse.Namespace) -> int:
    struct = parse_pdb(read_text_file(args.pdb), chain_id=args.chain)
    prof = analyze_structure(struct)
    dssp_counts = summarize_dssp(prof.dssp_codes)

    print(f"Structure: {struct.name} (Chain {struct.chain_id}, L={struct.length})")
    print(f"Sequence : {struct.sequence}")
    print(f"DSSP     : {prof.dssp_codes}")
    print(render_ramachandran_ascii(prof.phi_deg, prof.psi_deg))

    summary: dict[str, object] = {
        "length": struct.length,
        "dssp_counts": dssp_counts,
        "n_hydrogen_bonds": len(prof.hydrogen_bonds),
        "mean_rsa": float(np.mean(prof.rsa)),
        "buried_fraction": float(np.mean(prof.buried_mask)),
        "solvation_energy": prof.solvation_energy,
        "mean_rama_log_likelihood": float(np.mean(prof.ramachandran_log_likelihood)),
    }
    if args.reference_pdb:
        ref = parse_pdb(read_text_file(args.reference_pdb), chain_id=args.chain)
        n_cmp = min(struct.length, ref.length)
        _, _, _, rmsd = kabsch_superpose(struct.ca_coords[:n_cmp], ref.ca_coords[:n_cmp])
        tm = compute_tm_score(struct.ca_coords[:n_cmp], ref.ca_coords[:n_cmp])
        summary["kabsch_rmsd"] = rmsd
        summary["tm_score"] = tm
    print("Structure Summary:", json.dumps(summary, indent=2))
    return 0


def _cmd_phylo(args: argparse.Namespace) -> int:
    msa = parse_a3m(read_text_file(args.msa))
    seed_tree = (
        parse_newick(read_text_file(args.tree)) if args.tree else build_kimura_nj_tree(msa)
    )
    cfg = PhyloConfig(
        model_name=args.model,
        n_gamma_categories=args.gamma_cats,
        mcmc_steps=args.mcmc_steps,
        seed=args.seed,
    )
    ml_tree, alpha, ml_ll, mcmc_res, asr = infer_phylogeny_pipeline(
        msa, seed_tree, cfg, run_mcmc=(args.mcmc_steps > 0)
    )
    display_tree = mcmc_res.map_tree if mcmc_res is not None else ml_tree
    print(render_tree_ascii(display_tree))
    print("ML Newick:", write_newick(ml_tree, include_support=False))
    if mcmc_res is not None:
        print("Bayes MAP Newick:", write_newick(mcmc_res.map_tree, include_support=True))
    print(
        json.dumps(
            {
                "model": args.model,
                "alpha": alpha,
                "ml_log_likelihood": ml_ll,
                "tree_length": ml_tree.total_branch_length(),
                "asr_mean_confidence": asr.mean_posterior_confidence,
                "mcmc_ess": mcmc_res.ess_log_likelihood if mcmc_res else None,
                "mcmc_accept_rate": mcmc_res.acceptance_rate if mcmc_res else None,
            },
            indent=2,
        )
    )
    return 0


def _cmd_dynamics(args: argparse.Namespace) -> int:
    struct = parse_pdb(read_text_file(args.pdb), chain_id=args.chain)
    cfg = ElasticNetworkConfig(gnm_cutoff=args.gnm_cutoff, anm_cutoff=args.anm_cutoff)
    nma = run_elastic_network_analysis(struct, cfg)
    print(render_dynamics_profile_ascii(nma.gnm_bfactors, nma.hinge_residues))
    print(
        json.dumps(
            {
                "length": struct.length,
                "gnm_bfactor_pearson": nma.bfactor_pearson,
                "n_hinge_residues": len(nma.hinge_residues),
                "hinge_residues_1based": [h + 1 for h in nma.hinge_residues],
                "top5_anm_eigenvalues": [round(float(x), 4) for x in nma.anm_eigenvalues[:5]],
                "top5_collectivity": [round(float(x), 4) for x in nma.collectivity[:5]],
            },
            indent=2,
        )
    )
    return 0


def _cmd_fold(args: argparse.Namespace) -> int:
    msa = parse_a3m(read_text_file(args.msa))
    coev = compute_plmdca(msa, CoevolutionConfig(plm_maxiter=args.plm_maxiter))
    dssp = None
    ref_struct = None
    if args.pdb:
        ref_struct = parse_pdb(read_text_file(args.pdb))
        dssp = analyze_structure(ref_struct).dssp_codes

    fold_res = reconstruct_ca_fold(
        coev,
        dssp_codes=dssp,
        config=FoldingConfig(maxiter=args.fold_maxiter, seed=args.seed),
    )
    summary: dict[str, object] = {
        "length": int(fold_res.ca_coords.shape[0]),
        "total_energy": round(fold_res.total_energy, 4),
        "bond_energy": round(fold_res.bond_energy, 4),
        "local_ss_energy": round(fold_res.local_ss_energy, 4),
        "contact_energy": round(fold_res.contact_energy, 4),
        "clash_energy": round(fold_res.clash_energy, 4),
        "helix_handedness_deg": round(fold_res.helix_handedness_score, 2),
    }
    if ref_struct is not None:
        n_cmp = min(fold_res.ca_coords.shape[0], ref_struct.length)
        _, _, _, rmsd = kabsch_superpose(fold_res.ca_coords[:n_cmp], ref_struct.ca_coords[:n_cmp])
        tm = compute_tm_score(fold_res.ca_coords[:n_cmp], ref_struct.ca_coords[:n_cmp])
        summary["kabsch_rmsd"] = round(rmsd, 4)
        summary["tm_score"] = round(tm, 4)
    print("Contact-Restrained Folding Summary:", json.dumps(summary, indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    suite = generate_coupled_benchmark(
        length=args.length,
        n_taxa=args.n_taxa,
        n_coev_seqs=args.n_coev,
        model_name=args.model,
        seed=args.seed,
    )
    pipe = CladeFoldPipeline(
        coev_config=CoevolutionConfig(plm_maxiter=args.plm_maxiter),
        phylo_config=PhyloConfig(
            model_name=args.model, mcmc_steps=args.mcmc_steps, seed=args.seed
        ),
    )
    bundle = pipe.run(
        project_name="CladeFold_Synthetic_HelixBundle",
        structure=suite.structure,
        phylo_msa=suite.phylo_msa,
        coev_msa=suite.coev_msa,
        run_mcmc=(args.mcmc_steps > 0),
    )
    rf_dist, norm_rf = robinson_foulds_distance(suite.true_tree, bundle.ml_tree)
    print(bundle.ascii_dashboard)
    telemetry = bundle.to_dict()
    telemetry["robinson_foulds_vs_true"] = rf_dist
    telemetry["normalized_rf_vs_true"] = norm_rf
    print("\nPipeline Telemetry:", json.dumps(telemetry, indent=2))
    if args.json_out:
        write_json_report(telemetry, args.json_out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CladeFold multi-subcommand CLI parser."""
    parser = argparse.ArgumentParser(
        prog="cladefold",
        description=(
            "Structural proteomics, co-evolutionary contact inference, "
            "normal-mode dynamics, and Felsenstein phylogenetics."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_aln = sub.add_parser("align", help="Progressive & iterative Gotoh MSA from FASTA")
    p_aln.add_argument("--input", required=True, help="Unaligned FASTA input path")
    p_aln.add_argument("--output", help="Optional aligned FASTA output path")
    p_aln.add_argument("--matrix", default="BLOSUM62", choices=["BLOSUM62", "PAM250"])
    p_aln.add_argument("--gap-open", type=float, default=-10.0)
    p_aln.add_argument("--gap-extend", type=float, default=-1.0)
    p_aln.add_argument("--guide", default="nj", choices=["nj", "upgma"])
    p_aln.add_argument("--refine", type=int, default=2)
    p_aln.set_defaults(func=_cmd_align)

    p_co = sub.add_parser("coevolution", help="MI-APC / mfDCA / PLM-DCA contact inference")
    p_co.add_argument("--msa", required=True, help="Aligned FASTA or A3M path")
    p_co.add_argument("--pdb", help="Optional PDB reference for contact precision")
    p_co.add_argument("--method", default="plmdca", choices=["mi", "mfdca", "plmdca"])
    p_co.add_argument("--id-thresh", type=float, default=0.80)
    p_co.add_argument("--pseudocount", type=float, default=0.25)
    p_co.add_argument("--min-sep", type=int, default=6)
    p_co.add_argument("--maxiter", type=int, default=60)
    p_co.add_argument("--top-k", type=int, default=35)
    p_co.add_argument("--json-out", help="Optional JSON telemetry path")
    p_co.set_defaults(func=_cmd_coevolution)

    p_st = sub.add_parser("structure", help="DSSP, Shrake-Rupley SASA, Ramachandran & TM-score")
    p_st.add_argument("--pdb", required=True, help="Input PDB file")
    p_st.add_argument("--reference-pdb", help="Optional reference PDB for Kabsch/TM-score")
    p_st.add_argument("--chain", help="Optional chain ID")
    p_st.set_defaults(func=_cmd_structure)

    p_ph = sub.add_parser("phylo", help="Felsenstein ML, NNI, Bayesian MCMC & ASR")
    p_ph.add_argument("--msa", required=True, help="Aligned FASTA/A3M path")
    p_ph.add_argument("--tree", help="Optional Newick seed tree path")
    p_ph.add_argument("--model", default="WAG", choices=["JTT", "WAG", "LG"])
    p_ph.add_argument("--gamma-cats", type=int, default=4)
    p_ph.add_argument("--mcmc-steps", type=int, default=150)
    p_ph.add_argument("--seed", type=int, default=42)
    p_ph.set_defaults(func=_cmd_phylo)

    p_dy = sub.add_parser("dynamics", help="GNM & ANM elastic network normal-mode analysis")
    p_dy.add_argument("--pdb", required=True, help="Input PDB file")
    p_dy.add_argument("--chain", help="Optional chain ID")
    p_dy.add_argument("--gnm-cutoff", type=float, default=7.0)
    p_dy.add_argument("--anm-cutoff", type=float, default=13.0)
    p_dy.set_defaults(func=_cmd_dynamics)

    p_fd = sub.add_parser("fold", help="Contact-restrained 3D C-alpha distance-geometry folding")
    p_fd.add_argument("--msa", required=True, help="Aligned FASTA/A3M path")
    p_fd.add_argument("--pdb", help="Optional native PDB for DSSP restraints and TM-score")
    p_fd.add_argument("--plm-maxiter", type=int, default=40)
    p_fd.add_argument("--fold-maxiter", type=int, default=160)
    p_fd.add_argument("--seed", type=int, default=42)
    p_fd.set_defaults(func=_cmd_fold)

    p_demo = sub.add_parser("demo", help="Run full coupled synthetic benchmark pipeline")
    p_demo.add_argument("--length", type=int, default=32)
    p_demo.add_argument("--n-taxa", type=int, default=8)
    p_demo.add_argument("--n-coev", type=int, default=100)
    p_demo.add_argument("--model", default="WAG", choices=["JTT", "WAG", "LG"])
    p_demo.add_argument("--plm-maxiter", type=int, default=35)
    p_demo.add_argument("--mcmc-steps", type=int, default=80)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--json-out", help="Optional JSON telemetry path")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
