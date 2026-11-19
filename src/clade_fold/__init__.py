"""CladeFold: Structural proteomics, co-evolution, dynamics, and phylogenetics."""

from .coevolution import (
    CoevolutionConfig,
    compare_coevolution_methods,
    compute_mfdca,
    compute_mi_apc,
    compute_plmdca,
    evaluate_contact_predictions,
)
from .contact_folding import (
    FoldingConfig,
    FoldingResult,
    mds_distance_embedding,
    reconstruct_ca_fold,
)
from .elastic_dynamics import (
    ElasticNetworkConfig,
    analyze_evolution_dynamics_coupling,
    compute_deformation_overlap,
    run_elastic_network_analysis,
)
from .io_formats import (
    parse_a3m,
    parse_fasta,
    parse_newick,
    parse_pdb,
    write_fasta,
    write_newick,
    write_pdb,
)
from .msa_engine import (
    AlignmentConfig,
    align_pairwise_gotoh,
    build_progressive_msa,
    compute_conservation_profile,
)
from .phylogenetics import (
    McmcSummary,
    PhyloConfig,
    felsenstein_log_likelihood,
    infer_phylogeny_pipeline,
    reconstruct_ancestral_sequences,
    robinson_foulds_distance,
    run_bayesian_mcmc,
)
from .pipeline import CladeFoldPipeline, PipelineBundle, build_kimura_nj_tree
from .structural_geometry import (
    analyze_structure,
    assign_dssp_secondary_structure,
    compute_shrake_rupley_sasa,
    compute_tm_score,
    kabsch_superpose,
)
from .substitution_models import SubstitutionModel, discrete_gamma_rates
from .synthetic_benchmark import generate_coupled_benchmark, generate_helical_hairpin_fold
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

__version__ = "0.1.0"

__all__ = [
    "AlignmentConfig",
    "AncestralReconstructionResult",
    "CladeFoldPipeline",
    "CoevolutionConfig",
    "ContactMapResult",
    "ElasticNetworkConfig",
    "FoldingConfig",
    "FoldingResult",
    "McmcSummary",
    "MultipleSequenceAlignment",
    "NormalModeResult",
    "PhyloConfig",
    "PipelineBundle",
    "PipelineReport",
    "ProteinStructure",
    "SecondaryStructureProfile",
    "SubstitutionModel",
    "TreeNode",
    "align_pairwise_gotoh",
    "analyze_evolution_dynamics_coupling",
    "analyze_structure",
    "assign_dssp_secondary_structure",
    "build_kimura_nj_tree",
    "build_progressive_msa",
    "compare_coevolution_methods",
    "compute_conservation_profile",
    "compute_deformation_overlap",
    "compute_mfdca",
    "compute_mi_apc",
    "compute_plmdca",
    "compute_shrake_rupley_sasa",
    "compute_tm_score",
    "discrete_gamma_rates",
    "evaluate_contact_predictions",
    "felsenstein_log_likelihood",
    "generate_coupled_benchmark",
    "generate_helical_hairpin_fold",
    "infer_phylogeny_pipeline",
    "kabsch_superpose",
    "mds_distance_embedding",
    "parse_a3m",
    "parse_fasta",
    "parse_newick",
    "parse_pdb",
    "reconstruct_ancestral_sequences",
    "reconstruct_ca_fold",
    "robinson_foulds_distance",
    "run_bayesian_mcmc",
    "run_elastic_network_analysis",
    "write_fasta",
    "write_newick",
    "write_pdb",
]
