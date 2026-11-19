"""Domain types, biophysical constants, and invariant-checked containers for CladeFold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Canonical 20 standard amino acids in standard alphabetical 1-letter order
CANONICAL_AA: Tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
)
GAP_CHAR: str = "-"
UNKNOWN_AA: str = "X"
GAP_INDEX: int = 20
ALPHABET_WITH_GAP: Tuple[str, ...] = CANONICAL_AA + (GAP_CHAR,)

AA_TO_INDEX: Dict[str, int] = {aa: idx for idx, aa in enumerate(CANONICAL_AA)}
AA_TO_INDEX[GAP_CHAR] = GAP_INDEX
AA_TO_INDEX["."] = GAP_INDEX

ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    "X": "UNK",
}
THREE_TO_ONE: Dict[str, str] = {v: k for k, v in ONE_TO_THREE.items()}
THREE_TO_ONE.update({
    "MSE": "M",  # Selenomethionine
    "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "CYM": "C",
})

# Kyte-Doolittle hydropathy index (kcal/mol transfer scale, 1982)
KYTE_DOOLITTLE: Dict[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg normalized consensus hydrophobicity (1984) for solvation estimates
EISENBERG_HYDROPHOBICITY: Dict[str, float] = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}

# Tien et al. (2013) theoretical maximum SASA (Angstrom^2) in Gly-X-Gly tripeptide
MAX_SASA_TIEN: Dict[str, float] = {
    "A": 129.0, "C": 167.0, "D": 193.0, "E": 223.0, "F": 240.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "K": 236.0, "L": 201.0,
    "M": 224.0, "N": 195.0, "P": 159.0, "Q": 225.0, "R": 274.0,
    "S": 155.0, "T": 172.0, "V": 174.0, "W": 285.0, "Y": 263.0,
}

# Effective residue van der Waals radii for coarse/backbone-CB Shrake-Rupley SASA (Angstrom)
BACKBONE_VDW_RADII: Dict[str, float] = {
    "N": 1.55,
    "CA": 1.70,
    "C": 1.70,
    "O": 1.52,
    "CB": 1.85,
}

# Effective sidechain centroid radius boost (Angstrom) capturing heavy-atom steric envelope
RESIDUE_EFFECTIVE_CB_RADIUS: Dict[str, float] = {
    "A": 1.90, "C": 2.25, "D": 2.45, "E": 2.75, "F": 3.10,
    "G": 1.70, "H": 2.90, "I": 2.70, "K": 2.95, "L": 2.70,
    "M": 2.75, "N": 2.45, "P": 2.35, "Q": 2.75, "R": 3.20,
    "S": 2.15, "T": 2.35, "V": 2.45, "W": 3.45, "Y": 3.25,
}

# Monoisotopic / average residue molecular weight (Da)
RESIDUE_MASS_DA: Dict[str, float] = {
    "A": 71.08, "C": 103.14, "D": 115.09, "E": 129.12, "F": 147.18,
    "G": 57.05, "H": 137.14, "I": 113.16, "K": 128.17, "L": 113.16,
    "M": 131.20, "N": 114.10, "P": 97.12, "Q": 128.13, "R": 156.19,
    "S": 87.08, "T": 101.11, "V": 99.13, "W": 186.21, "Y": 163.18,
}


def encode_sequence(seq: str) -> IntArray:
    """Encode an aligned protein sequence into integers in [0, 20].

    Standard 20 amino acids map to 0..19; gaps ('-', '.') map to 20;
    ambiguous residues ('X', 'B', 'Z', '*') map deterministically to 20
    (treated as missing/unknown in likelihood and frequency sums).
    """
    arr = np.empty(len(seq), dtype=np.int64)
    for idx, ch in enumerate(seq.upper()):
        arr[idx] = AA_TO_INDEX.get(ch, GAP_INDEX)
    return arr


def decode_sequence(indices: Sequence[int] | IntArray) -> str:
    """Decode an integer sequence array in [0, 20] back into 1-letter symbols."""
    return "".join(
        ALPHABET_WITH_GAP[i] if 0 <= int(i) <= GAP_INDEX else UNKNOWN_AA
        for i in indices
    )


@dataclass(frozen=True)
class ResidueRecord:
    """Single amino acid residue with backbone and virtual/actual C-beta coordinates.

    Coordinates are in Angstroms. For Glycine, ``cb`` holds the tetrahedral
    virtual C-beta projection computed from (N, CA, C).
    """

    index: int
    resseq: int
    chain_id: str
    aa: str
    n: FloatArray
    ca: FloatArray
    c: FloatArray
    o: FloatArray
    cb: FloatArray
    b_factor: float = 20.0

    def __post_init__(self) -> None:
        for atom_name, vec in (
            ("n", self.n),
            ("ca", self.ca),
            ("c", self.c),
            ("o", self.o),
            ("cb", self.cb),
        ):
            if vec.shape != (3,):
                raise ValueError(f"Atom {atom_name} must have shape (3,), got {vec.shape}")
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"Atom {atom_name} contains non-finite coordinates")


@dataclass
class ProteinStructure:
    """Validated protein 3D structure container with vectorized backbone arrays."""

    name: str
    residues: List[ResidueRecord]
    chain_id: str = "A"
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.residues:
            raise ValueError("ProteinStructure must contain at least one residue")

    @property
    def length(self) -> int:
        return len(self.residues)

    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues)

    @property
    def ca_coords(self) -> FloatArray:
        return np.vstack([r.ca for r in self.residues]).astype(np.float64)

    @property
    def cb_coords(self) -> FloatArray:
        return np.vstack([r.cb for r in self.residues]).astype(np.float64)

    @property
    def n_coords(self) -> FloatArray:
        return np.vstack([r.n for r in self.residues]).astype(np.float64)

    @property
    def c_coords(self) -> FloatArray:
        return np.vstack([r.c for r in self.residues]).astype(np.float64)

    @property
    def o_coords(self) -> FloatArray:
        return np.vstack([r.o for r in self.residues]).astype(np.float64)

    @property
    def b_factors(self) -> FloatArray:
        return np.array([r.b_factor for r in self.residues], dtype=np.float64)

    def cb_distance_matrix(self) -> FloatArray:
        """Compute pairwise Euclidean C-beta distance matrix of shape (L, L) in Angstrom."""
        cb = self.cb_coords
        diff = cb[:, None, :] - cb[None, :, :]
        return np.asarray(np.sqrt(np.sum(diff * diff, axis=-1)), dtype=np.float64)

    def ca_distance_matrix(self) -> FloatArray:
        """Compute pairwise Euclidean C-alpha distance matrix of shape (L, L) in Angstrom."""
        ca = self.ca_coords
        diff = ca[:, None, :] - ca[None, :, :]
        return np.asarray(np.sqrt(np.sum(diff * diff, axis=-1)), dtype=np.float64)

    def contact_map(self, cutoff: float = 8.0, min_seq_sep: int = 6) -> NDArray[np.bool_]:
        """Return boolean 3D contact map where d_CB(i, j) <= cutoff and |i - j| >= min_seq_sep."""
        dist = self.cb_distance_matrix()
        L = self.length
        idx = np.arange(L)
        sep = np.abs(idx[:, None] - idx[None, :])
        return np.asarray((dist <= cutoff) & (sep >= min_seq_sep), dtype=np.bool_)


@dataclass
class MultipleSequenceAlignment:
    """Validated protein Multiple Sequence Alignment (MSA) with integer matrix encoding."""

    names: List[str]
    sequences: List[str]
    matrix: IntArray = field(init=False)
    weights: Optional[FloatArray] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.names or not self.sequences:
            raise ValueError("MSA cannot be empty")
        if len(self.names) != len(self.sequences):
            raise ValueError("names and sequences must have identical length")
        lengths = {len(s) for s in self.sequences}
        if len(lengths) != 1:
            raise ValueError(f"All aligned sequences must have equal length, got {sorted(lengths)}")
        if next(iter(lengths)) == 0:
            raise ValueError("MSA sequence length must be positive")
        self.matrix = np.vstack([encode_sequence(s) for s in self.sequences])
        if self.weights is not None:
            if self.weights.shape != (len(self.sequences),):
                raise ValueError("weights shape mismatch")

    @property
    def n_seqs(self) -> int:
        return len(self.sequences)

    @property
    def length(self) -> int:
        return int(self.matrix.shape[1])

    def gap_fractions(self) -> FloatArray:
        """Per-column gap fraction in [0, 1]."""
        return np.asarray(np.mean(self.matrix == GAP_INDEX, axis=0), dtype=np.float64)

    def consensus_sequence(self) -> str:
        """Compute weighted modal residue consensus sequence."""
        w = self.weights if self.weights is not None else np.ones(self.n_seqs, dtype=np.float64)
        out: List[str] = []
        for col in range(self.length):
            counts = np.zeros(21, dtype=np.float64)
            col_vals = self.matrix[:, col]
            for seq_i in range(self.n_seqs):
                counts[int(col_vals[seq_i])] += float(w[seq_i])
            best_aa = int(np.argmax(counts[:20]))
            if counts[GAP_INDEX] > counts[best_aa]:
                out.append(GAP_CHAR)
            else:
                out.append(CANONICAL_AA[best_aa])
        return "".join(out)


@dataclass
class TreeNode:
    """Node in a rooted or unrooted-bifurcating phylogenetic tree."""

    name: str = ""
    branch_length: float = 0.0
    support: Optional[float] = None
    children: List["TreeNode"] = field(default_factory=list)
    node_id: int = -1

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def postorder(self) -> List["TreeNode"]:
        """Return nodes in postorder (children before parent)."""
        order: List[TreeNode] = []
        stack: List[Tuple[TreeNode, bool]] = [(self, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                order.append(node)
            else:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))
        return order

    def preorder(self) -> List["TreeNode"]:
        """Return nodes in preorder (parent before children)."""
        order: List[TreeNode] = []
        stack: List[TreeNode] = [self]
        while stack:
            node = stack.pop()
            order.append(node)
            for child in reversed(node.children):
                stack.append(child)
        return order

    def leaves(self) -> List["TreeNode"]:
        """Return leaf nodes in left-to-right order."""
        return [n for n in self.preorder() if n.is_leaf]

    def assign_ids(self) -> int:
        """Assign contiguous 0..M-1 node IDs in postorder and return total node count."""
        nodes = self.postorder()
        for idx, node in enumerate(nodes):
            node.node_id = idx
        return len(nodes)

    def clone(self) -> "TreeNode":
        """Deep-copy the tree topology, branch lengths, and support values."""
        return TreeNode(
            name=self.name,
            branch_length=float(self.branch_length),
            support=self.support,
            children=[c.clone() for c in self.children],
            node_id=self.node_id,
        )

    def total_branch_length(self) -> float:
        """Sum of all non-root branch lengths in substitutions per site."""
        return float(sum(n.branch_length for n in self.postorder()[:-1]))


@dataclass
class ContactMapResult:
    """Output container for evolutionary co-variation and contact prediction."""

    method: str
    score_matrix: FloatArray
    raw_matrix: FloatArray
    top_pairs: List[Tuple[int, int, float]]
    m_eff: float
    precision_at_l5: Optional[float] = None
    precision_at_l2: Optional[float] = None
    precision_at_l: Optional[float] = None
    auroc: Optional[float] = None


@dataclass
class SecondaryStructureProfile:
    """Per-residue DSSP, Ramachandran, and Shrake-Rupley SASA structural profile."""

    dssp_codes: str
    phi_deg: FloatArray
    psi_deg: FloatArray
    omega_deg: FloatArray
    ramachandran_zones: List[str]
    ramachandran_log_likelihood: FloatArray
    sasa: FloatArray
    rsa: FloatArray
    buried_mask: NDArray[np.bool_]
    solvation_energy: float
    hydrogen_bonds: List[Tuple[int, int, float]]


@dataclass
class NormalModeResult:
    """GNM and ANM elastic network normal-mode dynamics output."""

    gnm_eigenvalues: FloatArray
    gnm_eigenvectors: FloatArray
    gnm_bfactors: FloatArray
    bfactor_pearson: Optional[float]
    hinge_residues: List[int]
    anm_eigenvalues: FloatArray
    anm_eigenvectors: FloatArray
    anm_msf: FloatArray
    cross_correlation: FloatArray
    collectivity: FloatArray


@dataclass
class AncestralReconstructionResult:
    """Marginal and joint ancestral sequence reconstruction across internal tree nodes."""

    marginal_sequences: Dict[str, str]
    marginal_posteriors: Dict[str, FloatArray]
    joint_sequences: Dict[str, str]
    mean_posterior_confidence: float


@dataclass
class PipelineReport:
    """Comprehensive telemetry report from an integrated CladeFold analysis run."""

    project_name: str
    n_taxa: int
    alignment_length: int
    m_eff: float
    substitution_model: str
    ml_log_likelihood: float
    tree_length: float
    newick_ml: str
    newick_bayes: Optional[str]
    contact_precision_l5: Optional[float]
    contact_precision_l: Optional[float]
    dssp_summary: Dict[str, int]
    mean_rsa: float
    solvation_energy: float
    gnm_bfactor_pearson: Optional[float]
    evolution_dynamics_spearman: float
    asr_mean_confidence: float
    digest_sha256: str
