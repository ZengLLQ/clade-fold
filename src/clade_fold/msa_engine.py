"""Progressive and iterative Multiple Sequence Alignment engine with Gotoh affine gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .substitution_models import get_scoring_matrix
from .types import (
    CANONICAL_AA,
    FloatArray,
    GAP_CHAR,
    GAP_INDEX,
    IntArray,
    MultipleSequenceAlignment,
    TreeNode,
    encode_sequence,
)


@dataclass(frozen=True)
class AlignmentConfig:
    """Parameters governing progressive and iterative MSA construction."""

    matrix_name: str = "BLOSUM62"
    gap_open: float = -10.0
    gap_extend: float = -1.0
    terminal_gap_factor: float = 0.5
    guide_method: str = "nj"
    refinement_rounds: int = 2


def compute_henikoff_weights(msa_matrix: IntArray) -> FloatArray:
    """Compute Henikoff & Henikoff (1994) position-based sequence weights.

    Downweights redundant sequences in over-sampled clades so that profile
    alignment and conservation metrics reflect evolutionary diversity rather
    than sequencing bias. Weights are normalized to sum to ``N_seqs``.
    """
    n_seqs, length = msa_matrix.shape
    if n_seqs == 1:
        return np.ones(1, dtype=np.float64)
    weights = np.zeros(n_seqs, dtype=np.float64)
    valid_cols = 0
    for col in range(length):
        col_vals = msa_matrix[:, col]
        non_gap = col_vals[col_vals < 20]
        if non_gap.size == 0:
            continue
        unique_aa, counts = np.unique(non_gap, return_counts=True)
        r = float(unique_aa.size)
        count_map = {int(aa): float(cnt) for aa, cnt in zip(unique_aa, counts)}
        has_gap = bool(np.any(col_vals == GAP_INDEX))
        denom_r = r + (1.0 if has_gap else 0.0)
        gap_cnt = float(np.sum(col_vals == GAP_INDEX))
        for i in range(n_seqs):
            aa = int(col_vals[i])
            if aa < 20:
                weights[i] += 1.0 / (denom_r * count_map[aa])
            else:
                weights[i] += 1.0 / (denom_r * max(1.0, gap_cnt))
        valid_cols += 1

    if valid_cols == 0 or float(np.sum(weights)) <= 0.0:
        return np.ones(n_seqs, dtype=np.float64)
    weights *= float(n_seqs) / float(np.sum(weights))
    return weights


def _kmer_set(seq: str, k: int = 3) -> set[str]:
    cleaned = seq.replace(GAP_CHAR, "").upper()
    if len(cleaned) < k:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + k] for i in range(len(cleaned) - k + 1)}


def kmer_distance_matrix(sequences: Sequence[str], k: int = 3) -> FloatArray:
    """Compute fast pairwise k-mer Jaccard-transformed evolutionary distance matrix."""
    n = len(sequences)
    kmers = [_kmer_set(s, k=k) for s in sequences]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = kmers[i], kmers[j]
            union_sz = len(a | b)
            sim = float(len(a & b)) / float(union_sz) if union_sz > 0 else 0.0
            d = -np.log(max(1e-4, 0.05 + 0.95 * sim))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def kimura_distance_from_aligned(seq_a: str, seq_b: str) -> Tuple[float, float]:
    """Compute fractional identity and Kimura (1983) distance between two aligned strings."""
    matches = 0
    aligned_pos = 0
    for ca, cb in zip(seq_a, seq_b):
        if ca == GAP_CHAR or cb == GAP_CHAR:
            continue
        aligned_pos += 1
        if ca.upper() == cb.upper():
            matches += 1
    if aligned_pos == 0:
        return 0.0, 4.0
    identity = float(matches) / float(aligned_pos)
    p = 1.0 - identity
    inner = 1.0 - p - 0.2 * p * p
    if inner <= 0.02:
        return identity, 4.0
    return identity, float(-np.log(inner))


def neighbor_joining(dist_matrix: FloatArray, labels: Sequence[str]) -> TreeNode:
    """Reconstruct an unrooted-then-midpoint-rooted Neighbor-Joining tree (Saitou & Nei 1987)."""
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot build tree from zero taxa")
    if n == 1:
        root = TreeNode(name=labels[0], branch_length=0.0)
        root.assign_ids()
        return root
    if n == 2:
        half = max(1e-4, 0.5 * float(dist_matrix[0, 1]))
        root = TreeNode(
            children=[
                TreeNode(name=labels[0], branch_length=half),
                TreeNode(name=labels[1], branch_length=half),
            ]
        )
        root.assign_ids()
        return root

    D = dist_matrix.astype(np.float64).copy()
    active_nodes: List[TreeNode] = [TreeNode(name=lbl) for lbl in labels]
    active_ids = list(range(n))

    while len(active_nodes) > 2:
        m = len(active_nodes)
        sub = D[np.ix_(active_ids, active_ids)]
        row_sums = np.sum(sub, axis=1)
        # Q_ij = (m - 2) * D_ij - row_sum_i - row_sum_j
        Q = (m - 2.0) * sub - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)
        flat_idx = int(np.argmin(Q))
        i, j = divmod(flat_idx, m)
        if i > j:
            i, j = j, i

        d_ij = float(sub[i, j])
        delta = (float(row_sums[i]) - float(row_sums[j])) / (2.0 * (m - 2.0))
        len_i = max(1e-5, 0.5 * d_ij + delta)
        len_j = max(1e-5, d_ij - len_i)

        node_i = active_nodes[i]
        node_j = active_nodes[j]
        node_i.branch_length = len_i
        node_j.branch_length = len_j
        parent = TreeNode(children=[node_i, node_j])

        # Compute distances from new parent to remaining k
        idx_i = active_ids[i]
        idx_j = active_ids[j]
        new_dists = np.zeros(D.shape[0], dtype=np.float64)
        for k_pos, orig_k in enumerate(active_ids):
            if k_pos in (i, j):
                continue
            new_dists[orig_k] = max(0.0, 0.5 * (D[idx_i, orig_k] + D[idx_j, orig_k] - d_ij))

        # Expand D by one row/column for parent
        new_idx = D.shape[0]
        expanded = np.zeros((new_idx + 1, new_idx + 1), dtype=np.float64)
        expanded[:new_idx, :new_idx] = D
        expanded[new_idx, :new_idx] = new_dists
        expanded[:new_idx, new_idx] = new_dists
        D = expanded

        # Remove j then i
        for drop in (j, i):
            active_nodes.pop(drop)
            active_ids.pop(drop)
        active_nodes.append(parent)
        active_ids.append(new_idx)

    # Join final two active nodes into rooted bifurcation
    last_dist = max(2e-5, float(D[active_ids[0], active_ids[1]]))
    active_nodes[0].branch_length = 0.5 * last_dist
    active_nodes[1].branch_length = 0.5 * last_dist
    root = TreeNode(children=[active_nodes[0], active_nodes[1]])
    root.assign_ids()
    return root


def upgma_tree(dist_matrix: FloatArray, labels: Sequence[str]) -> TreeNode:
    """Construct an ultrametric UPGMA guide tree from a pairwise distance matrix."""
    n = len(labels)
    if n == 1:
        root = TreeNode(name=labels[0])
        root.assign_ids()
        return root

    D = dist_matrix.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)
    nodes: List[TreeNode] = [TreeNode(name=lbl) for lbl in labels]
    heights: List[float] = [0.0 for _ in range(n)]
    sizes: List[int] = [1 for _ in range(n)]

    while len(nodes) > 1:
        m = len(nodes)
        sub = D[:m, :m]
        flat_idx = int(np.argmin(sub))
        i, j = divmod(flat_idx, m)
        if i > j:
            i, j = j, i
        new_h = 0.5 * float(sub[i, j])
        nodes[i].branch_length = max(1e-5, new_h - heights[i])
        nodes[j].branch_length = max(1e-5, new_h - heights[j])
        merged = TreeNode(children=[nodes[i], nodes[j]])
        new_size = sizes[i] + sizes[j]

        # Compute weighted UPGMA distance to remaining clusters
        rem = [k for k in range(m) if k not in (i, j)]
        new_row = [
            (sizes[i] * D[i, k] + sizes[j] * D[j, k]) / float(new_size)
            for k in rem
        ]
        # Rebuild reduced matrix
        next_m = len(rem) + 1
        next_D = np.full((next_m, next_m), np.inf, dtype=np.float64)
        for a_idx, orig_a in enumerate(rem):
            for b_idx in range(a_idx + 1, len(rem)):
                orig_b = rem[b_idx]
                next_D[a_idx, b_idx] = D[orig_a, orig_b]
                next_D[b_idx, a_idx] = D[orig_a, orig_b]
            next_D[a_idx, next_m - 1] = new_row[a_idx]
            next_D[next_m - 1, a_idx] = new_row[a_idx]

        nodes = [nodes[k] for k in rem] + [merged]
        heights = [heights[k] for k in rem] + [new_h]
        sizes = [sizes[k] for k in rem] + [new_size]
        D = next_D

    nodes[0].assign_ids()
    return nodes[0]


@dataclass
class _SequenceProfile:
    """Internal weighted amino-acid frequency profile for profile-profile Gotoh DP."""

    names: List[str]
    sequences: List[str]
    weights: FloatArray
    aa_freqs: FloatArray  # shape (L, 20), normalized per non-gap residue
    non_gap_frac: FloatArray  # shape (L,), fraction of weight that is non-gap

    @classmethod
    def from_sequences(
        cls,
        names: List[str],
        sequences: List[str],
        weights: Optional[FloatArray] = None,
    ) -> "_SequenceProfile":
        mat = np.vstack([encode_sequence(s) for s in sequences])
        n_seqs, length = mat.shape
        if weights is None:
            w = compute_henikoff_weights(mat)
        else:
            w = weights.astype(np.float64).copy()
        w_norm = w / max(1e-12, float(np.sum(w)))

        aa_freqs = np.zeros((length, 20), dtype=np.float64)
        non_gap = np.zeros(length, dtype=np.float64)
        for col in range(length):
            col_vals = mat[:, col]
            tot_aa = 0.0
            for s_i in range(n_seqs):
                aa = int(col_vals[s_i])
                if aa < 20:
                    aa_freqs[col, aa] += w_norm[s_i]
                    tot_aa += w_norm[s_i]
            non_gap[col] = tot_aa
            if tot_aa > 1e-12:
                aa_freqs[col, :] /= tot_aa
            else:
                aa_freqs[col, :] = 0.05
        return cls(
            names=list(names),
            sequences=list(sequences),
            weights=w,
            aa_freqs=aa_freqs,
            non_gap_frac=non_gap,
        )


def _align_two_profiles(
    prof_a: _SequenceProfile,
    prof_b: _SequenceProfile,
    scoring_mat: FloatArray,
    gap_open: float,
    gap_extend: float,
    terminal_gap_factor: float = 0.5,
) -> Tuple[_SequenceProfile, float]:
    """Gotoh (1982) affine-gap global profile-profile alignment with O(L1 * L2) DP."""
    la = prof_a.aa_freqs.shape[0]
    lb = prof_b.aa_freqs.shape[0]

    # Precompute all-pairs sum-of-pairs column substitution matrix S[i, j]
    # S[i, j] = (non_gap_a[i] * non_gap_b[j]) * (f_a[i] @ scoring_mat @ f_b[j])
    pair_score = (prof_a.aa_freqs @ scoring_mat) @ prof_b.aa_freqs.T
    ng_weight = prof_a.non_gap_frac[:, None] * prof_b.non_gap_frac[None, :]
    col_scores = pair_score * ng_weight

    # Gap penalty modulated by column occupancy (so already-gappy columns attract gaps)
    gap_scale_a = 0.5 + 0.5 * prof_a.non_gap_frac
    gap_scale_b = 0.5 + 0.5 * prof_b.non_gap_frac

    neg_inf = -1e15
    M = np.full((la + 1, lb + 1), neg_inf, dtype=np.float64)
    Ix = np.full((la + 1, lb + 1), neg_inf, dtype=np.float64)  # gap in B (advance A)
    Iy = np.full((la + 1, lb + 1), neg_inf, dtype=np.float64)  # gap in A (advance B)

    # State traceback codes: 0 = M, 1 = Ix, 2 = Iy
    ptr_M = np.zeros((la + 1, lb + 1), dtype=np.int8)
    ptr_Ix = np.zeros((la + 1, lb + 1), dtype=np.int8)
    ptr_Iy = np.zeros((la + 1, lb + 1), dtype=np.int8)

    M[0, 0] = 0.0
    t_open = gap_open * terminal_gap_factor
    t_ext = gap_extend * terminal_gap_factor

    for i in range(1, la + 1):
        Ix[i, 0] = t_open + (i - 1) * t_ext
        ptr_Ix[i, 0] = 1 if i > 1 else 0
    for j in range(1, lb + 1):
        Iy[0, j] = t_open + (j - 1) * t_ext
        ptr_Iy[0, j] = 2 if j > 1 else 0

    for i in range(1, la + 1):
        row_scores = col_scores[i - 1]
        sa = float(gap_scale_a[i - 1])
        is_last_i = i == la
        go_i = gap_open * (terminal_gap_factor if is_last_i else 1.0)
        ge_i = gap_extend * (terminal_gap_factor if is_last_i else 1.0)

        for j in range(1, lb + 1):
            sb = float(gap_scale_b[j - 1])
            is_last_j = j == lb
            go_j = gap_open * (terminal_gap_factor if is_last_j else 1.0)
            ge_j = gap_extend * (terminal_gap_factor if is_last_j else 1.0)

            # Match/mismatch state M[i, j]
            m_cands = (M[i - 1, j - 1], Ix[i - 1, j - 1], Iy[i - 1, j - 1])
            best_m_prev = 0
            best_m_val = m_cands[0]
            if m_cands[1] > best_m_val:
                best_m_val = m_cands[1]
                best_m_prev = 1
            if m_cands[2] > best_m_val:
                best_m_val = m_cands[2]
                best_m_prev = 2
            M[i, j] = best_m_val + float(row_scores[j - 1])
            ptr_M[i, j] = best_m_prev

            # Gap in B (consume residue i from A): Ix[i, j]
            open_ix = M[i - 1, j] + go_j * sa
            ext_ix = Ix[i - 1, j] + ge_j * sa
            cross_ix = Iy[i - 1, j] + go_j * sa
            if ext_ix >= open_ix and ext_ix >= cross_ix:
                Ix[i, j] = ext_ix
                ptr_Ix[i, j] = 1
            elif open_ix >= cross_ix:
                Ix[i, j] = open_ix
                ptr_Ix[i, j] = 0
            else:
                Ix[i, j] = cross_ix
                ptr_Ix[i, j] = 2

            # Gap in A (consume residue j from B): Iy[i, j]
            open_iy = M[i, j - 1] + go_i * sb
            ext_iy = Iy[i, j - 1] + ge_i * sb
            cross_iy = Ix[i, j - 1] + go_i * sb
            if ext_iy >= open_iy and ext_iy >= cross_iy:
                Iy[i, j] = ext_iy
                ptr_Iy[i, j] = 2
            elif open_iy >= cross_iy:
                Iy[i, j] = open_iy
                ptr_Iy[i, j] = 0
            else:
                Iy[i, j] = cross_iy
                ptr_Iy[i, j] = 1

    # Select terminal state
    end_vals = (M[la, lb], Ix[la, lb], Iy[la, lb])
    state = int(np.argmax(end_vals))
    best_score = float(end_vals[state])

    # Backtrack alignment path
    cols_a: List[Optional[int]] = []
    cols_b: List[Optional[int]] = []
    i, j = la, lb
    while i > 0 or j > 0:
        if state == 0:
            prev = int(ptr_M[i, j])
            i -= 1
            j -= 1
            cols_a.append(i)
            cols_b.append(j)
            state = prev
        elif state == 1:
            prev = int(ptr_Ix[i, j])
            i -= 1
            cols_a.append(i)
            cols_b.append(None)
            state = prev
        else:
            prev = int(ptr_Iy[i, j])
            j -= 1
            cols_a.append(None)
            cols_b.append(j)
            state = prev

    cols_a.reverse()
    cols_b.reverse()

    aligned_seqs: List[str] = []
    for seq in prof_a.sequences:
        aligned_seqs.append("".join(seq[idx] if idx is not None else GAP_CHAR for idx in cols_a))
    for seq in prof_b.sequences:
        aligned_seqs.append("".join(seq[idx] if idx is not None else GAP_CHAR for idx in cols_b))

    merged_names = prof_a.names + prof_b.names
    merged = _SequenceProfile.from_sequences(merged_names, aligned_seqs)
    return merged, best_score


def align_pairwise_gotoh(
    seq_a: str,
    seq_b: str,
    matrix_name: str = "BLOSUM62",
    gap_open: float = -10.0,
    gap_extend: float = -1.0,
) -> Tuple[str, str, float, float]:
    """Perform Gotoh affine-gap global pairwise alignment between two protein sequences.

    Returns:
        (aligned_a, aligned_b, alignment_score, fractional_identity)
    """
    smat = get_scoring_matrix(matrix_name)
    pa = _SequenceProfile.from_sequences(["a"], [seq_a.replace(GAP_CHAR, "").upper()])
    pb = _SequenceProfile.from_sequences(["b"], [seq_b.replace(GAP_CHAR, "").upper()])
    merged, score = _align_two_profiles(pa, pb, smat, gap_open, gap_extend)
    aln_a, aln_b = merged.sequences
    ident, _ = kimura_distance_from_aligned(aln_a, aln_b)
    return aln_a, aln_b, score, ident


def _strip_all_gap_columns(sequences: List[str]) -> List[str]:
    """Remove columns where every sequence has a gap."""
    if not sequences:
        return sequences
    length = len(sequences[0])
    keep = [col for col in range(length) if any(seq[col] != GAP_CHAR for seq in sequences)]
    if not keep:
        return sequences
    return ["".join(seq[c] for c in keep) for seq in sequences]


def _sum_of_pairs_score(
    msa: MultipleSequenceAlignment, scoring_mat: FloatArray, gap_open: float, gap_extend: float
) -> float:
    """Compute exact weighted sum-of-pairs objective for an MSA."""
    mat = msa.matrix
    n_seqs, length = mat.shape
    total = 0.0
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            in_gap = False
            for c in range(length):
                a = int(mat[i, c])
                b = int(mat[j, c])
                if a == GAP_INDEX and b == GAP_INDEX:
                    continue
                if a == GAP_INDEX or b == GAP_INDEX:
                    total += gap_extend if in_gap else gap_open
                    in_gap = True
                else:
                    total += float(scoring_mat[a, b])
                    in_gap = False
    return total


def build_progressive_msa(
    records: Sequence[Tuple[str, str]],
    config: Optional[AlignmentConfig] = None,
) -> MultipleSequenceAlignment:
    """Build a MultipleSequenceAlignment via guide-tree progressive Gotoh + iterative refinement.

    Pipeline:
      1. Compute pairwise k-mer Jaccard evolutionary distance matrix.
      2. Construct Neighbor-Joining (or UPGMA) guide tree.
      3. Traverse tree postorder, aligning child profiles via Gotoh affine-gap DP.
      4. Refine via leave-one-out / bipartite split re-alignment optimizing Sum-of-Pairs.
    """
    if not records:
        raise ValueError("Cannot align empty sequence set")
    cfg = config or AlignmentConfig()
    smat = get_scoring_matrix(cfg.matrix_name)
    names = [r[0] for r in records]
    raw_seqs = [r[1].replace(GAP_CHAR, "").upper() for r in records]

    if len(records) == 1:
        return MultipleSequenceAlignment(names=names, sequences=raw_seqs)

    dist = kmer_distance_matrix(raw_seqs, k=3)
    if cfg.guide_method.lower() == "upgma":
        guide = upgma_tree(dist, names)
    else:
        guide = neighbor_joining(dist, names)

    name_to_seq = dict(zip(names, raw_seqs))
    node_profiles: Dict[int, _SequenceProfile] = {}

    for node in guide.postorder():
        if node.is_leaf:
            node_profiles[node.node_id] = _SequenceProfile.from_sequences(
                [node.name], [name_to_seq[node.name]]
            )
        else:
            acc = node_profiles[node.children[0].node_id]
            for child in node.children[1:]:
                acc, _ = _align_two_profiles(
                    acc,
                    node_profiles[child.node_id],
                    smat,
                    cfg.gap_open,
                    cfg.gap_extend,
                    cfg.terminal_gap_factor,
                )
            node_profiles[node.node_id] = acc

    root_prof = node_profiles[guide.node_id]
    # Restore original input sequence ordering
    seq_by_name = dict(zip(root_prof.names, root_prof.sequences))
    ordered_seqs = [seq_by_name[n] for n in names]
    msa = MultipleSequenceAlignment(names=names, sequences=ordered_seqs)

    # Iterative leave-one-out refinement
    if len(names) >= 3 and cfg.refinement_rounds > 0:
        best_sp = _sum_of_pairs_score(msa, smat, cfg.gap_open, cfg.gap_extend)
        for _ in range(cfg.refinement_rounds):
            improved = False
            for idx in range(len(names)):
                other_names = [n for k, n in enumerate(names) if k != idx]
                other_seqs = _strip_all_gap_columns(
                    [s for k, s in enumerate(msa.sequences) if k != idx]
                )
                prof_rest = _SequenceProfile.from_sequences(other_names, other_seqs)
                prof_single = _SequenceProfile.from_sequences([names[idx]], [raw_seqs[idx]])
                merged, _ = _align_two_profiles(
                    prof_rest,
                    prof_single,
                    smat,
                    cfg.gap_open,
                    cfg.gap_extend,
                    cfg.terminal_gap_factor,
                )
                cand_map = dict(zip(merged.names, merged.sequences))
                cand_seqs = _strip_all_gap_columns([cand_map[n] for n in names])
                cand_msa = MultipleSequenceAlignment(names=names, sequences=cand_seqs)
                cand_sp = _sum_of_pairs_score(cand_msa, smat, cfg.gap_open, cfg.gap_extend)
                if cand_sp > best_sp + 1e-6:
                    msa = cand_msa
                    best_sp = cand_sp
                    improved = True
            if not improved:
                break

    final_weights = compute_henikoff_weights(msa.matrix)
    return MultipleSequenceAlignment(
        names=msa.names,
        sequences=msa.sequences,
        weights=final_weights,
        metadata={"matrix": cfg.matrix_name, "guide": cfg.guide_method},
    )


def compute_conservation_profile(msa: MultipleSequenceAlignment) -> FloatArray:
    """Compute per-column normalized Shannon conservation in [0, 1] (1 = fully conserved)."""
    weights = (
        msa.weights if msa.weights is not None else compute_henikoff_weights(msa.matrix)
    )
    n_seqs, length = msa.matrix.shape
    scores = np.zeros(length, dtype=np.float64)
    log20 = float(np.log(20.0))

    for col in range(length):
        counts = np.zeros(20, dtype=np.float64)
        total_w = 0.0
        gap_w = 0.0
        for i in range(n_seqs):
            aa = int(msa.matrix[i, col])
            w = float(weights[i])
            total_w += w
            if aa < 20:
                counts[aa] += w
            else:
                gap_w += w
        aa_w = total_w - gap_w
        if aa_w <= 1e-12:
            scores[col] = 0.0
            continue
        probs = counts / aa_w
        nz = probs[probs > 1e-12]
        entropy = -float(np.sum(nz * np.log(nz)))
        gap_penalty = aa_w / max(1e-12, total_w)
        scores[col] = max(0.0, min(1.0, (1.0 - entropy / log20) * gap_penalty))
    return scores


def trim_gappy_columns(
    msa: MultipleSequenceAlignment, max_gap_fraction: float = 0.5
) -> MultipleSequenceAlignment:
    """Filter MSA columns whose gap fraction exceeds ``max_gap_fraction``."""
    gfrac = msa.gap_fractions()
    keep = np.where(gfrac <= max_gap_fraction)[0]
    if keep.size == 0:
        raise ValueError("All MSA columns exceed max_gap_fraction threshold")
    trimmed = ["".join(seq[int(c)] for c in keep) for seq in msa.sequences]
    return MultipleSequenceAlignment(names=list(msa.names), sequences=trimmed)


def pairwise_identity_matrix(msa: MultipleSequenceAlignment) -> FloatArray:
    """Compute N x N pairwise sequence identity matrix over aligned positions."""
    mat = msa.matrix
    n = msa.n_seqs
    ident = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            valid = (mat[i] < 20) & (mat[j] < 20)
            denom = int(np.sum(valid))
            if denom == 0:
                val = 0.0
            else:
                val = float(np.sum(mat[i, valid] == mat[j, valid])) / float(denom)
            ident[i, j] = val
            ident[j, i] = val
    return ident


def summarise_msa(msa: MultipleSequenceAlignment) -> Dict[str, float]:
    """Compute summary alignment depth, gap, identity, and conservation statistics."""
    cons = compute_conservation_profile(msa)
    ident = pairwise_identity_matrix(msa)
    triu = ident[np.triu_indices(msa.n_seqs, k=1)] if msa.n_seqs > 1 else np.array([1.0])
    return {
        "n_seqs": float(msa.n_seqs),
        "length": float(msa.length),
        "mean_gap_fraction": float(np.mean(msa.gap_fractions())),
        "mean_pairwise_identity": float(np.mean(triu)),
        "mean_conservation": float(np.mean(cons)),
        "highly_conserved_sites": float(np.sum(cons >= 0.8)),
        "canonical_aa_count": float(len(CANONICAL_AA)),
    }
