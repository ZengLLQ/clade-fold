"""Example 04: GNM/ANM normal modes, evolution-dynamics coupling, and contact-restrained folding."""

from __future__ import annotations

from clade_fold.ascii_viz import render_dynamics_profile_ascii
from clade_fold.coevolution import CoevolutionConfig, compute_coupling_centrality, compute_plmdca
from clade_fold.contact_folding import FoldingConfig, reconstruct_ca_fold
from clade_fold.elastic_dynamics import (
    ElasticNetworkConfig,
    analyze_evolution_dynamics_coupling,
    run_elastic_network_analysis,
)
from clade_fold.msa_engine import compute_conservation_profile
from clade_fold.structural_geometry import analyze_structure, compute_tm_score, kabsch_superpose
from clade_fold.synthetic_benchmark import generate_coupled_benchmark


def main() -> None:
    suite = generate_coupled_benchmark(length=30, n_taxa=8, n_coev_seqs=140, seed=19)
    prof = analyze_structure(suite.structure)
    nma = run_elastic_network_analysis(suite.structure, ElasticNetworkConfig())

    print(render_dynamics_profile_ascii(nma.gnm_bfactors, nma.hinge_residues))
    print(f"GNM vs B-factor Pearson r : {nma.bfactor_pearson}")
    print(f"Hinge residues (1-based)  : {[h + 1 for h in nma.hinge_residues]}")
    print(f"Top 3 ANM Eigenvalues     : {[round(float(v), 4) for v in nma.anm_eigenvalues[:3]]}")

    coev = compute_plmdca(suite.coev_msa, CoevolutionConfig(plm_maxiter=40))
    cent = compute_coupling_centrality(coev)
    cons = compute_conservation_profile(suite.coev_msa)
    coupling = analyze_evolution_dynamics_coupling(nma, cent, cons, prof.rsa)
    print("Evolution-Dynamics Coupling:", coupling)

    fold_res = reconstruct_ca_fold(
        coev,
        dssp_codes=prof.dssp_codes,
        config=FoldingConfig(maxiter=160, n_restarts=2, seed=42),
    )
    _, _, _, rmsd = kabsch_superpose(fold_res.ca_coords, suite.structure.ca_coords)
    tm = compute_tm_score(fold_res.ca_coords, suite.structure.ca_coords)
    print(
        f"\nContact-Restrained 3D Fold -> Energy={fold_res.total_energy:.2f}, "
        f"RMSD={rmsd:.2f} A, TM-score={tm:.3f}, "
        f"HelixHandedness={fold_res.helix_handedness_score:.1f} deg"
    )


if __name__ == "__main__":
    main()
