"""Integrated end-to-end structural proteomics, co-evolution, dynamics, and phylogeny pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
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
    compute_coupling_centrality,
    compute_plmdca,
    evaluate_contact_predictions,
)
from .elastic_dynamics import (
    ElasticNetworkConfig,
    analyze_evolution_dynamics_coupling,
    run_elastic_network_analysis,
)
from .io_formats import compute_msa_digest, write_newick
from .msa_engine import (
    AlignmentConfig,
    build_progressive_msa,
    compute_conservation_profile,
    kmer_distance_matrix,
    kimura_distance_from_aligned,
    neighbor_joining,
)
from .phylogenetics import (
    McmcSummary,
    PhyloConfig,
    infer_phylogeny_pipeline,
)
from .structural_geometry import analyze_structure, summarize_dssp
from .types import (
    AncestralReconstructionResult,
    ContactMapResult,
    MultipleSequenceAlignment,
    NormalModeResult,
    PipelineReport,
    ProteinStructure,
    SecondaryStructureProfile,
    TreeNode,
)


@dataclass
class PipelineBundle:
    """Complete output bundle containing all subsystem artifacts and summary telemetry."""

    report: PipelineReport
    phylo_msa: MultipleSequenceAlignment
    coev_msa: MultipleSequenceAlignment
    ml_tree: TreeNode
    mcmc_summary: Optional[McmcSummary]
    asr_result: AncestralReconstructionResult
    contact_result: ContactMapResult
    structure_profile: SecondaryStructureProfile
    nma_result: NormalModeResult
    coupling_metrics: Dict[str, float]
    ascii_dashboard: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scalar telemetry, Newick trees, and top contact predictions to JSON dict."""
        return {
            "project_name": self.report.project_name,
            "n_taxa": self.report.n_taxa,
            "alignment_length": self.report.alignment_length,
            "m_eff": self.report.m_eff,
            "substitution_model": self.report.substitution_model,
            "ml_log_likelihood": self.report.ml_log_likelihood,
            "tree_length": self.report.tree_length,
            "newick_ml": self.report.newick_ml,
            "newick_bayes": self.report.newick_bayes,
            "contact_precision_l5": self.report.contact_precision_l5,
            "contact_precision_l": self.report.contact_precision_l,
            "contact_auroc": self.contact_result.auroc,
            "dssp_summary": self.report.dssp_summary,
            "mean_rsa": self.report.mean_rsa,
            "solvation_energy": self.report.solvation_energy,
            "gnm_bfactor_pearson": self.report.gnm_bfactor_pearson,
            "evolution_dynamics_spearman": self.report.evolution_dynamics_spearman,
            "coupling_metrics": self.coupling_metrics,
            "asr_mean_confidence": self.report.asr_mean_confidence,
            "hinge_residues": self.nma_result.hinge_residues,
            "top_contacts": [
                {"i": i, "j": j, "score": s} for i, j, s in self.contact_result.top_pairs[:15]
            ],
            "digest_sha256": self.report.digest_sha256,
        }


def build_kimura_nj_tree(msa: MultipleSequenceAlignment) -> TreeNode:
    """Build a Neighbor-Joining seed tree using pairwise Kimura protein distances."""
    n = msa.n_seqs
    if n < 2:
        return neighbor_joining(np.zeros((1, 1), dtype=np.float64), msa.names)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            _, kd = kimura_distance_from_aligned(msa.sequences[i], msa.sequences[j])
            dist[i, j] = kd
            dist[j, i] = kd
    if float(np.max(dist)) <= 0.0:
        dist = kmer_distance_matrix(msa.sequences)
    return neighbor_joining(dist, msa.names)


class CladeFoldPipeline:
    """Orchestrator uniting MSA, DCA contact inference, 3D geometry, Felsenstein phylo, and NMA."""

    def __init__(
        self,
        align_config: Optional[AlignmentConfig] = None,
        coev_config: Optional[CoevolutionConfig] = None,
        phylo_config: Optional[PhyloConfig] = None,
        nma_config: Optional[ElasticNetworkConfig] = None,
    ) -> None:
        self.align_config = align_config or AlignmentConfig()
        self.coev_config = coev_config or CoevolutionConfig()
        self.phylo_config = phylo_config or PhyloConfig()
        self.nma_config = nma_config or ElasticNetworkConfig()

    def run(
        self,
        project_name: str,
        structure: ProteinStructure,
        phylo_msa: Optional[MultipleSequenceAlignment] = None,
        unaligned_records: Optional[list[tuple[str, str]]] = None,
        coev_msa: Optional[MultipleSequenceAlignment] = None,
        run_mcmc: bool = True,
    ) -> PipelineBundle:
        """Execute the full multi-subsystem CladeFold analysis workflow."""
        if phylo_msa is None:
            if not unaligned_records:
                raise ValueError("Provide either phylo_msa or unaligned_records")
            phylo_msa = build_progressive_msa(unaligned_records, self.align_config)

        active_coev_msa = coev_msa if coev_msa is not None else phylo_msa
        phylo_cons = compute_conservation_profile(phylo_msa)
        coev_cons = (
            phylo_cons
            if active_coev_msa is phylo_msa
            else compute_conservation_profile(active_coev_msa)
        )

        # 1. Co-evolutionary contact inference (PLM-DCA)
        contact_res = compute_plmdca(active_coev_msa, self.coev_config)
        contact_res = evaluate_contact_predictions(
            contact_res, structure, min_seq_sep=self.coev_config.min_seq_sep
        )
        centrality = compute_coupling_centrality(
            contact_res, min_seq_sep=self.coev_config.min_seq_sep
        )

        # 2. 3D structural geometry (DSSP, Ramachandran, Shrake-Rupley SASA)
        struct_prof = analyze_structure(structure)
        dssp_summary = summarize_dssp(struct_prof.dssp_codes)

        # 3. Phylogenetic inference (Kimura NJ -> Felsenstein ML/NNI -> Bayesian MCMC -> ASR)
        seed_tree = build_kimura_nj_tree(phylo_msa)
        ml_tree, _, ml_ll, mcmc_res, asr_res = infer_phylogeny_pipeline(
            phylo_msa, seed_tree, self.phylo_config, run_mcmc=run_mcmc
        )

        # 4. Normal-mode elastic network dynamics & evolution-dynamics coupling
        nma_res = run_elastic_network_analysis(structure, self.nma_config)
        coupling_metrics = analyze_evolution_dynamics_coupling(
            nma_res, centrality, coev_cons, struct_prof.rsa
        )

        newick_ml = write_newick(ml_tree, include_support=False)
        newick_bayes = (
            write_newick(mcmc_res.map_tree, include_support=True)
            if mcmc_res is not None
            else None
        )
        digest = compute_msa_digest(phylo_msa)

        # 5. Pure 7-bit ASCII integrated dashboard
        true_cmap = structure.contact_map(cutoff=8.0, min_seq_sep=self.coev_config.min_seq_sep)
        sections = [
            f"=== CladeFold Pipeline Dashboard: {project_name} ===",
            render_msa_alignment_ascii(
                phylo_msa, conservation=phylo_cons, dssp_codes=struct_prof.dssp_codes, max_seqs=8
            ),
            render_contact_map_ascii(
                contact_res.score_matrix,
                true_contacts=true_cmap,
                top_k=max(10, structure.length // 2),
                min_seq_sep=self.coev_config.min_seq_sep,
            ),
            render_tree_ascii(mcmc_res.map_tree if mcmc_res is not None else ml_tree),
            render_ramachandran_ascii(struct_prof.phi_deg, struct_prof.psi_deg),
            render_dynamics_profile_ascii(nma_res.gnm_bfactors, nma_res.hinge_residues),
        ]
        dashboard = "\n\n".join(sections)

        report = PipelineReport(
            project_name=project_name,
            n_taxa=phylo_msa.n_seqs,
            alignment_length=phylo_msa.length,
            m_eff=contact_res.m_eff,
            substitution_model=self.phylo_config.model_name,
            ml_log_likelihood=ml_ll,
            tree_length=ml_tree.total_branch_length(),
            newick_ml=newick_ml,
            newick_bayes=newick_bayes,
            contact_precision_l5=contact_res.precision_at_l5,
            contact_precision_l=contact_res.precision_at_l,
            dssp_summary=dssp_summary,
            mean_rsa=float(np.mean(struct_prof.rsa)),
            solvation_energy=struct_prof.solvation_energy,
            gnm_bfactor_pearson=nma_res.bfactor_pearson,
            evolution_dynamics_spearman=coupling_metrics["spearman_conservation_vs_bfactor"],
            asr_mean_confidence=asr_res.mean_posterior_confidence,
            digest_sha256=digest,
        )

        return PipelineBundle(
            report=report,
            phylo_msa=phylo_msa,
            coev_msa=active_coev_msa,
            ml_tree=ml_tree,
            mcmc_summary=mcmc_res,
            asr_result=asr_res,
            contact_result=contact_res,
            structure_profile=struct_prof,
            nma_result=nma_res,
            coupling_metrics=coupling_metrics,
            ascii_dashboard=dashboard,
        )
