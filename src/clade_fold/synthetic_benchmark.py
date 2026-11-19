"""NeRF 3D protein fold generator and contact-coupled phylogenetic sequence simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .elastic_dynamics import build_gnm_kirchhoff
from .io_formats import compute_virtual_cb
from .substitution_models import SubstitutionModel
from .types import (
    CANONICAL_AA,
    FloatArray,
    KYTE_DOOLITTLE,
    MultipleSequenceAlignment,
    ProteinStructure,
    ResidueRecord,
    TreeNode,
)


@dataclass
class SyntheticBenchmarkSuite:
    """Self-consistent 3D protein fold, ground-truth tree, phylogeny MSA, and deep Potts MSA."""

    structure: ProteinStructure
    true_tree: TreeNode
    phylo_msa: MultipleSequenceAlignment
    coev_msa: MultipleSequenceAlignment
    unaligned_sequences: List[Tuple[str, str]]
    ancestral_sequences: Dict[str, str]


def nerf_place_atom(
    p_a: FloatArray,
    p_b: FloatArray,
    p_c: FloatArray,
    bond_len: float,
    bond_angle_deg: float,
    dihedral_deg: float,
) -> FloatArray:
    """Place atom D from preceding atoms (A, B, C) via Natural Extension Reference Frame (NeRF)."""
    bc = p_c - p_b
    bc_u = bc / max(1e-12, float(np.linalg.norm(bc)))
    ab = p_b - p_a
    n = np.cross(ab, bc_u)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        # Degenerate collinear fallback
        n = np.cross(np.array([0.0, 0.0, 1.0]), bc_u)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            n = np.array([1.0, 0.0, 0.0])
            n_norm = 1.0
    n_u = n / n_norm
    m_u = np.cross(n_u, bc_u)

    theta = np.radians(180.0 - bond_angle_deg)
    chi = np.radians(dihedral_deg)
    local = np.array([
        bond_len * np.cos(theta),
        bond_len * np.sin(theta) * np.cos(chi),
        bond_len * np.sin(theta) * np.sin(chi),
    ], dtype=np.float64)
    return np.asarray(p_c + local[0] * bc_u + local[1] * m_u + local[2] * n_u, dtype=np.float64)


def _build_backbone_from_dihedrals(
    sequence: str,
    phi_deg: FloatArray,
    psi_deg: FloatArray,
    omega_deg: Optional[FloatArray] = None,
    name: str = "synthetic_fold",
) -> ProteinStructure:
    """Construct an ideal peptide backbone (N, CA, C, O, CB) from (phi, psi, omega)."""
    L = len(sequence)
    if omega_deg is None:
        omega_deg = np.full(L, 180.0, dtype=np.float64)

    # Ideal Engh & Huber (1991) peptide bond lengths & angles
    d_n_ca = 1.458
    d_ca_c = 1.525
    d_c_n = 1.329
    d_c_o = 1.231

    ang_c_n_ca = 121.7
    ang_n_ca_c = 111.2
    ang_ca_c_n = 116.2
    ang_ca_c_o = 120.8

    N_pts = np.zeros((L, 3), dtype=np.float64)
    CA_pts = np.zeros((L, 3), dtype=np.float64)
    C_pts = np.zeros((L, 3), dtype=np.float64)
    O_pts = np.zeros((L, 3), dtype=np.float64)

    # Seed residue 0
    N_pts[0] = np.array([0.0, 0.0, 0.0])
    CA_pts[0] = np.array([d_n_ca, 0.0, 0.0])
    ang0 = np.radians(180.0 - ang_n_ca_c)
    C_pts[0] = CA_pts[0] + np.array([d_ca_c * np.cos(ang0), d_ca_c * np.sin(ang0), 0.0])

    for i in range(1, L):
        N_pts[i] = nerf_place_atom(
            N_pts[i - 1], CA_pts[i - 1], C_pts[i - 1],
            d_c_n, ang_ca_c_n, float(psi_deg[i - 1])
        )
        # Carbonyl O of residue i-1 lies trans to N_i around CA_{i-1}-C_{i-1}
        O_pts[i - 1] = nerf_place_atom(
            N_pts[i - 1], CA_pts[i - 1], C_pts[i - 1],
            d_c_o, ang_ca_c_o, float(psi_deg[i - 1]) + 180.0
        )
        CA_pts[i] = nerf_place_atom(
            CA_pts[i - 1], C_pts[i - 1], N_pts[i],
            d_n_ca, ang_c_n_ca, float(omega_deg[i])
        )
        C_pts[i] = nerf_place_atom(
            C_pts[i - 1], N_pts[i], CA_pts[i],
            d_ca_c, ang_n_ca_c, float(phi_deg[i])
        )

    # Terminal O_{L-1}
    O_pts[L - 1] = nerf_place_atom(
        N_pts[L - 1], CA_pts[L - 1], C_pts[L - 1],
        d_c_o, ang_ca_c_o, 0.0
    )

    # Compute realistic GNM-derived crystallographic B-factors + mild noise
    Gamma = build_gnm_kirchhoff(CA_pts, cutoff=7.0, gamma=1.0)
    evals, evecs = np.linalg.eigh(Gamma)
    inv_lam = np.where(evals > 1e-8, 1.0 / np.maximum(1e-8, evals), 0.0)
    msf = np.sum((evecs * evecs) * inv_lam[None, :], axis=1)
    bfactors = 12.0 + 35.0 * (msf / max(1e-8, float(np.max(msf))))

    residues: List[ResidueRecord] = []
    for i in range(L):
        cb = compute_virtual_cb(N_pts[i], CA_pts[i], C_pts[i])
        residues.append(
            ResidueRecord(
                index=i,
                resseq=i + 1,
                chain_id="A",
                aa=sequence[i],
                n=N_pts[i],
                ca=CA_pts[i],
                c=C_pts[i],
                o=O_pts[i],
                cb=cb,
                b_factor=float(bfactors[i]),
            )
        )
    return ProteinStructure(name=name, residues=residues, chain_id="A")


def generate_helical_hairpin_fold(length: int = 40, seed: int = 42) -> ProteinStructure:
    """Generate a compact 2- or 3-helix bundle 3D protein fold via NeRF kinematics.

    Alternates right-handed alpha-helices (phi=-63, psi=-43) with tight beta-turns
    so distant helical faces pack within 5-8 Angstrom C-beta contact distance.
    """
    if length < 18:
        raise ValueError("Helical hairpin length must be >= 18")
    rng = np.random.default_rng(seed)

    # Partition into 2 helices (for L <= 38) or 3 helices (for L > 38) with 5-res turns
    n_helices = 2 if length <= 38 else 3
    turn_len = 5
    helix_total = length - (n_helices - 1) * turn_len
    h_len = helix_total // n_helices

    phi = np.full(length, -63.0, dtype=np.float64)
    psi = np.full(length, -43.0, dtype=np.float64)

    # Place clash-free antiparallel hairpin loop at helix junctions (d_CB_min ~ 4.9 A)
    pos = h_len
    for _ in range(n_helices - 1):
        turn_phis = [-90.0, -70.0, 70.0, -100.0, -75.0]
        turn_psis = [130.0, 130.0, 130.0, 140.0, 150.0]
        for k in range(turn_len):
            if pos + k < length:
                phi[pos + k] = turn_phis[k]
                psi[pos + k] = turn_psis[k]
        pos += turn_len + h_len

    # Slight natural thermal perturbation (+/- 2 deg)
    phi += rng.normal(0.0, 1.8, size=length)
    psi += rng.normal(0.0, 1.8, size=length)

    # Assign amphipathic sequence matching helical periodicity
    aa_hydro = ["A", "L", "I", "V", "F", "M"]
    aa_polar = ["E", "K", "D", "R", "Q", "S", "N", "T"]
    seq_chars: List[str] = []
    for i in range(length):
        if i % 7 in (0, 3, 4):
            seq_chars.append(aa_hydro[int(rng.integers(0, len(aa_hydro)))])
        else:
            seq_chars.append(aa_polar[int(rng.integers(0, len(aa_polar)))])

    return _build_backbone_from_dihedrals(
        "".join(seq_chars), phi, psi, name="synthetic_helix_bundle"
    )


def generate_beta_hairpin_fold(length: int = 32, seed: int = 11) -> ProteinStructure:
    """Generate an antiparallel beta-sheet hairpin + flanking helix 3D protein structure."""
    if length < 20:
        raise ValueError("Beta hairpin length must be >= 20")
    rng = np.random.default_rng(seed)

    phi = np.full(length, -135.0, dtype=np.float64)
    psi = np.full(length, 135.0, dtype=np.float64)

    # Classic 2-residue tight hairpin turn forming antiparallel beta-sheet H-bonds
    mid = length // 3
    phi[mid] = 55.0
    psi[mid] = -125.0
    phi[mid + 1] = -90.0
    psi[mid + 1] = 10.0

    # C-terminal alpha-helix packing against sheet
    helix_start = (2 * length) // 3
    for i in range(helix_start, length):
        phi[i] = -63.0 + float(rng.normal(0.0, 1.5))
        psi[i] = -43.0 + float(rng.normal(0.0, 1.5))

    aa_pool = ["V", "I", "T", "Y", "S", "E", "K", "L", "A", "N"]
    seq = "".join(aa_pool[int(rng.integers(0, len(aa_pool)))] for _ in range(length))
    return _build_backbone_from_dihedrals(seq, phi, psi, name="synthetic_alpha_beta")


def generate_random_phylo_tree(
    n_taxa: int = 10, seed: int = 42, min_branch: float = 0.06, max_branch: float = 0.30
) -> TreeNode:
    """Generate a random bifurcating phylogenetic tree with labeled taxa ``Taxon_0..N-1``."""
    if n_taxa < 2:
        raise ValueError("n_taxa must be >= 2")
    rng = np.random.default_rng(seed)
    nodes = [
        TreeNode(name=f"Taxon_{i}", branch_length=float(rng.uniform(min_branch, max_branch)))
        for i in range(n_taxa)
    ]
    while len(nodes) > 1:
        idx_a, idx_b = sorted(rng.choice(len(nodes), size=2, replace=False).tolist())
        b = nodes.pop(idx_b)
        a = nodes.pop(idx_a)
        parent = TreeNode(
            children=[a, b],
            branch_length=float(rng.uniform(min_branch, max_branch)),
        )
        nodes.append(parent)
    root = nodes[0]
    root.branch_length = 0.0
    root.assign_ids()
    return root


def _build_contact_potts_couplings(
    structure: ProteinStructure, coupling_strength: float = 2.4
) -> Tuple[FloatArray, List[Tuple[int, int]]]:
    """Construct physical 20x20 Potts couplings for 3D C-beta contact pairs."""
    L = structure.length
    cmap = structure.contact_map(cutoff=8.5, min_seq_sep=6)
    contact_pairs = [(i, j) for i in range(L) for j in range(i + 6, L) if cmap[i, j]]

    # Biochemical complementarity matrix C_aa(a, b) encouraging co-evolving pairs:
    # opposite-charge salt bridges + knobs-into-holes (large <-> small) steric packing
    pos_aa = {CANONICAL_AA.index(x) for x in ("K", "R", "H")}
    neg_aa = {CANONICAL_AA.index(x) for x in ("D", "E")}
    large_hydro = {CANONICAL_AA.index(x) for x in ("F", "W", "Y", "I", "L", "M")}
    small_aa = {CANONICAL_AA.index(x) for x in ("A", "G", "S", "T", "V")}
    C_aa = np.zeros((20, 20), dtype=np.float64)
    for a in range(20):
        for b in range(20):
            if (a in pos_aa and b in neg_aa) or (a in neg_aa and b in pos_aa):
                C_aa[a, b] = coupling_strength * 1.35
            elif (a in large_hydro and b in small_aa) or (a in small_aa and b in large_hydro):
                C_aa[a, b] = coupling_strength * 0.95
            elif (a in pos_aa and b in pos_aa) or (a in neg_aa and b in neg_aa):
                C_aa[a, b] = -coupling_strength * 1.1
            elif a in large_hydro and b in large_hydro:
                C_aa[a, b] = -coupling_strength * 0.65
            else:
                C_aa[a, b] = -0.10 * coupling_strength
    return C_aa, contact_pairs


def simulate_potts_msa(
    structure: ProteinStructure,
    n_seqs: int = 120,
    coupling_strength: float = 2.6,
    burnin_sweeps: int = 12,
    sample_sweeps: int = 3,
    seed: int = 42,
) -> MultipleSequenceAlignment:
    """Sample a deep equilibrium MSA from a structure-derived 3D-contact Potts Hamiltonian.

    Site fields h_i(a) favor hydrophobic residues at buried core sites and polar
    residues at solvent-exposed sites; contact couplings J_{ij}(a, b) enforce
    salt-bridge and hydrophobic complementarity across true 3D contacts.
    """
    rng = np.random.default_rng(seed)
    L = structure.length
    C_aa, contact_pairs = _build_contact_potts_couplings(structure, coupling_strength)

    # Adjacency list for fast local Gibbs conditionals
    nbrs: List[List[int]] = [[] for _ in range(L)]
    for i, j in contact_pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)

    # Single-site fields biased by burial (C-beta neighbor count)
    dist = structure.cb_distance_matrix()
    burial = np.sum(dist < 9.0, axis=1)
    burial_norm = (burial - np.mean(burial)) / max(1e-6, float(np.std(burial)))
    kd_vec = np.array([KYTE_DOOLITTLE[aa] for aa in CANONICAL_AA], dtype=np.float64) / 4.5
    h_fields = 0.45 * burial_norm[:, None] * kd_vec[None, :]  # (L, 20)

    def _gibbs_sweep(state: np.ndarray) -> None:
        for i in rng.permutation(L):
            logits = h_fields[i].copy()
            for j in nbrs[i]:
                logits += C_aa[:, int(state[j])]
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= np.sum(probs)
            state[i] = int(rng.choice(20, p=probs))

    names: List[str] = []
    seqs: List[str] = []
    # Run 4 independent chains for diversity
    n_chains = 4
    per_chain = (n_seqs + n_chains - 1) // n_chains
    seq_id = 0
    for _ in range(n_chains):
        state = rng.integers(0, 20, size=L, dtype=np.int64)
        for _ in range(burnin_sweeps):
            _gibbs_sweep(state)
        for _ in range(per_chain):
            if seq_id >= n_seqs:
                break
            for _ in range(sample_sweeps):
                _gibbs_sweep(state)
            names.append(f"Homolog_{seq_id:03d}")
            seqs.append("".join(CANONICAL_AA[int(a)] for a in state))
            seq_id += 1

    return MultipleSequenceAlignment(names=names, sequences=seqs)


def simulate_tree_evolution(
    tree: TreeNode,
    length: int,
    model_name: str = "WAG",
    alpha: float = 1.0,
    seed: int = 42,
    introduce_indels: bool = False,
) -> Tuple[MultipleSequenceAlignment, Dict[str, str], List[Tuple[str, str]]]:
    """Simulate site-wise CTMC sequence evolution along a phylogenetic tree.

    Returns:
        (aligned_msa, ancestral_sequences_by_node_label, unaligned_fasta_records)
    """
    rng = np.random.default_rng(seed)
    model = SubstitutionModel.from_name(model_name)

    # Site-specific Gamma rate multipliers
    site_rates = rng.gamma(shape=alpha, scale=1.0 / alpha, size=length)

    # Sample root sequence from equilibrium distribution pi
    root_state = rng.choice(20, size=length, p=model.pi)
    node_states: Dict[int, np.ndarray] = {tree.node_id: root_state}

    ancestral_seqs: Dict[str, str] = {}
    int_counter = 0
    for node in tree.preorder():
        if not node.is_leaf:
            lbl = node.name if node.name else f"Node_{int_counter}"
            int_counter += 1
            ancestral_seqs[lbl] = "".join(CANONICAL_AA[int(a)] for a in node_states[node.node_id])

        for ch in node.children:
            parent_seq = node_states[node.node_id]
            ch_seq = np.empty(length, dtype=np.int64)
            # Group transition matrices slightly or compute exact P(rate_s * t)
            for s in range(length):
                P_s = model.transition_matrix(ch.branch_length * float(site_rates[s]))
                ch_seq[s] = int(rng.choice(20, p=P_s[int(parent_seq[s])]))
            node_states[ch.node_id] = ch_seq

    leaf_names: List[str] = []
    aligned_seqs: List[str] = []
    unaligned: List[Tuple[str, str]] = []

    for lf in tree.leaves():
        seq_str = "".join(CANONICAL_AA[int(a)] for a in node_states[lf.node_id])
        leaf_names.append(lf.name)
        if introduce_indels and length > 16:
            # Introduce a small deterministic-rate single-residue deletion in ~20% of leaves
            if rng.random() < 0.25:
                del_pos = int(rng.integers(3, length - 3))
                aln = seq_str[:del_pos] + "-" + seq_str[del_pos + 1:]
                aligned_seqs.append(aln)
                unaligned.append((lf.name, aln.replace("-", "")))
                continue
        aligned_seqs.append(seq_str)
        unaligned.append((lf.name, seq_str))

    msa = MultipleSequenceAlignment(names=leaf_names, sequences=aligned_seqs)
    return msa, ancestral_seqs, unaligned


def generate_coupled_benchmark(
    length: int = 36,
    n_taxa: int = 10,
    n_coev_seqs: int = 120,
    model_name: str = "WAG",
    seed: int = 42,
) -> SyntheticBenchmarkSuite:
    """Create a complete end-to-end synthetic structural proteomics & phylogeny benchmark."""
    structure = generate_helical_hairpin_fold(length=length, seed=seed)
    tree = generate_random_phylo_tree(n_taxa=n_taxa, seed=seed + 1)
    phylo_msa, anc_seqs, unaligned = simulate_tree_evolution(
        tree, length=length, model_name=model_name, alpha=1.2, seed=seed + 2
    )
    coev_msa = simulate_potts_msa(
        structure, n_seqs=n_coev_seqs, coupling_strength=2.8, seed=seed + 3
    )
    return SyntheticBenchmarkSuite(
        structure=structure,
        true_tree=tree,
        phylo_msa=phylo_msa,
        coev_msa=coev_msa,
        unaligned_sequences=unaligned,
        ancestral_sequences=anc_seqs,
    )
