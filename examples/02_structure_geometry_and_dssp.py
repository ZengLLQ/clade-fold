"""Example 02: 3D structural geometry, DSSP H-bonds, Shrake-Rupley SASA, and Kabsch TM-score."""

from __future__ import annotations

import numpy as np

from clade_fold.ascii_viz import render_ramachandran_ascii
from clade_fold.io_formats import parse_pdb, write_pdb
from clade_fold.structural_geometry import (
    analyze_structure,
    compute_tm_score,
    kabsch_superpose,
    summarize_dssp,
)
from clade_fold.synthetic_benchmark import (
    generate_beta_hairpin_fold,
    generate_helical_hairpin_fold,
)


def main() -> None:
    helix_fold = generate_helical_hairpin_fold(length=34, seed=7)
    pdb_text = write_pdb(helix_fold)
    reloaded = parse_pdb(pdb_text, name="HelixBundle")

    prof = analyze_structure(reloaded)
    print(f"Helix Bundle Length : {reloaded.length}")
    print(f"Sequence            : {reloaded.sequence}")
    print(f"DSSP Secondary Str. : {prof.dssp_codes}")
    print(f"DSSP Counts         : {summarize_dssp(prof.dssp_codes)}")
    print(f"Backbone H-Bonds    : {len(prof.hydrogen_bonds)}")
    print(f"Mean RSA            : {float(np.mean(prof.rsa)):.3f}")
    print(f"Buried Fraction     : {float(np.mean(prof.buried_mask)):.3f}")
    print(f"Eisenberg Solvation : {prof.solvation_energy:.2f}")
    print("\n" + render_ramachandran_ascii(prof.phi_deg, prof.psi_deg))

    # Rigid-body superposition check
    ca = reloaded.ca_coords
    rotated = ca @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T + 10.0
    _, _, _, rmsd = kabsch_superpose(rotated, ca)
    tm = compute_tm_score(rotated, ca)
    print(f"\nKabsch RMSD after 90-deg rotation: {rmsd:.3e} A | TM-score: {tm:.6f}")

    beta_fold = generate_beta_hairpin_fold(length=28, seed=11)
    beta_prof = analyze_structure(beta_fold)
    print(f"\nAlpha/Beta Fold DSSP: {beta_prof.dssp_codes}")
    print(f"Alpha/Beta Counts   : {summarize_dssp(beta_prof.dssp_codes)}")


if __name__ == "__main__":
    main()
