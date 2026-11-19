"""End-to-end tests for CladeFoldPipeline, pure 7-bit ASCII visualizers, and all CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

from clade_fold.ascii_viz import (
    render_contact_map_ascii,
    render_dynamics_profile_ascii,
    render_msa_alignment_ascii,
    render_ramachandran_ascii,
    render_tree_ascii,
)
from clade_fold.cli import main as cli_main
from clade_fold.coevolution import CoevolutionConfig
from clade_fold.io_formats import write_fasta, write_newick, write_pdb
from clade_fold.phylogenetics import PhyloConfig
from clade_fold.pipeline import CladeFoldPipeline
from clade_fold.synthetic_benchmark import generate_coupled_benchmark


def _is_pure_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def test_full_pipeline_and_pure_ascii_dashboard() -> None:
    suite = generate_coupled_benchmark(length=28, n_taxa=6, n_coev_seqs=80, seed=17)
    pipe = CladeFoldPipeline(
        coev_config=CoevolutionConfig(plm_maxiter=25),
        phylo_config=PhyloConfig(mcmc_steps=40, seed=17),
    )
    bundle = pipe.run(
        project_name="UnitTestRun",
        structure=suite.structure,
        unaligned_records=suite.unaligned_sequences,
        coev_msa=suite.coev_msa,
        run_mcmc=True,
    )
    assert _is_pure_ascii(bundle.ascii_dashboard)
    data = bundle.to_dict()
    assert data["n_taxa"] == 6
    assert data["alignment_length"] == 28
    assert data["ml_log_likelihood"] < 0.0
    assert data["asr_mean_confidence"] > 0.45
    assert len(data["digest_sha256"]) == 64

    # Individual visualizers strictly 7-bit ASCII
    assert _is_pure_ascii(render_tree_ascii(bundle.ml_tree))
    assert _is_pure_ascii(render_msa_alignment_ascii(bundle.phylo_msa))
    assert _is_pure_ascii(render_contact_map_ascii(bundle.contact_result.score_matrix))
    assert _is_pure_ascii(
        render_ramachandran_ascii(
            bundle.structure_profile.phi_deg, bundle.structure_profile.psi_deg
        )
    )
    assert _is_pure_ascii(
        render_dynamics_profile_ascii(
            bundle.nma_result.gnm_bfactors, bundle.nma_result.hinge_residues
        )
    )


def test_all_cli_subcommands_end_to_end(tmp_path: Path, capsys: object) -> None:
    suite = generate_coupled_benchmark(length=24, n_taxa=5, n_coev_seqs=60, seed=9)
    unaligned_fa = tmp_path / "unaligned.fasta"
    aligned_fa = tmp_path / "aligned.fasta"
    coev_fa = tmp_path / "coev.fasta"
    pdb_path = tmp_path / "fold.pdb"
    tree_path = tmp_path / "seed.nwk"
    demo_json = tmp_path / "demo.json"

    unaligned_fa.write_text(write_fasta(suite.unaligned_sequences), encoding="utf-8")
    coev_fa.write_text(
        write_fasta(list(zip(suite.coev_msa.names, suite.coev_msa.sequences))), encoding="utf-8"
    )
    pdb_path.write_text(write_pdb(suite.structure), encoding="utf-8")
    tree_path.write_text(write_newick(suite.true_tree), encoding="utf-8")

    assert cli_main(["align", "--input", str(unaligned_fa), "--output", str(aligned_fa)]) == 0
    assert aligned_fa.exists()

    assert (
        cli_main([
            "coevolution", "--msa", str(coev_fa), "--pdb", str(pdb_path),
            "--method", "mfdca", "--top-k", "15",
        ])
        == 0
    )
    assert cli_main(["structure", "--pdb", str(pdb_path), "--reference-pdb", str(pdb_path)]) == 0
    assert (
        cli_main([
            "phylo", "--msa", str(aligned_fa), "--tree", str(tree_path),
            "--mcmc-steps", "30",
        ])
        == 0
    )
    assert cli_main(["dynamics", "--pdb", str(pdb_path)]) == 0
    assert (
        cli_main([
            "fold", "--msa", str(coev_fa), "--pdb", str(pdb_path),
            "--plm-maxiter", "20", "--fold-maxiter", "60",
        ])
        == 0
    )
    assert (
        cli_main([
            "demo", "--length", "24", "--n-taxa", "5", "--n-coev", "50",
            "--plm-maxiter", "15", "--mcmc-steps", "25", "--json-out", str(demo_json),
        ])
        == 0
    )
    assert demo_json.exists()
    payload = json.loads(demo_json.read_text(encoding="utf-8"))
    assert payload["n_taxa"] == 5
