"""Felsenstein pruning likelihood, ML NNI tree search, Bayesian MCMC, and ASR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import minimize_scalar

from .substitution_models import SubstitutionModel, discrete_gamma_rates
from .types import (
    AncestralReconstructionResult,
    CANONICAL_AA,
    FloatArray,
    IntArray,
    MultipleSequenceAlignment,
    TreeNode,
)


@dataclass(frozen=True)
class PhyloConfig:
    """Configuration for Maximum-Likelihood and Bayesian phylogenetic inference."""

    model_name: str = "WAG"
    n_gamma_categories: int = 4
    gamma_alpha: float = 1.0
    ml_rounds: int = 2
    nni_max_rounds: int = 2
    mcmc_steps: int = 200
    burnin_fraction: float = 0.25
    branch_prior_rate: float = 10.0
    seed: int = 42


@dataclass
class McmcSummary:
    """Summary of Bayesian phylogenetic MCMC posterior sampling."""

    map_tree: TreeNode
    map_log_posterior: float
    mean_log_likelihood: float
    mean_tree_length: float
    mean_alpha: float
    ess_log_likelihood: float
    acceptance_rate: float
    clade_posteriors: Dict[FrozenSet[str], float]
    log_posterior_trace: List[float]


def _align_msa_to_tree(
    msa: MultipleSequenceAlignment, tree: TreeNode
) -> Dict[str, IntArray]:
    name_map = {name: msa.matrix[idx] for idx, name in enumerate(msa.names)}
    leaves = tree.leaves()
    missing = [lf.name for lf in leaves if lf.name not in name_map]
    if missing:
        raise ValueError(f"Tree leaves missing from MSA: {missing}")
    return name_map


def felsenstein_log_likelihood(
    tree: TreeNode,
    msa: MultipleSequenceAlignment,
    model: SubstitutionModel,
    alpha: float = 1.0,
    n_gamma: int = 4,
) -> float:
    """Compute exact log-likelihood log P(MSA | T, t, alpha, model) via Felsenstein pruning.

    Uses discrete Gamma rate heterogeneity across ``n_gamma`` categories and
    per-node log-rescaling to remain numerically stable for deep trees and long alignments.
    """
    tree.assign_ids()
    name_map = _align_msa_to_tree(msa, tree)
    rates = discrete_gamma_rates(alpha, n_categories=n_gamma)
    K = len(rates)
    L = msa.length

    cond_lik: Dict[int, FloatArray] = {}
    log_scales: Dict[int, FloatArray] = {}

    for node in tree.postorder():
        if node.is_leaf:
            seq_idx = name_map[node.name]
            arr = np.zeros((K, L, 20), dtype=np.float64)
            for s in range(L):
                aa = int(seq_idx[s])
                if aa < 20:
                    arr[:, s, aa] = 1.0
                else:
                    arr[:, s, :] = 1.0
            cond_lik[node.node_id] = arr
            log_scales[node.node_id] = np.zeros((K, L), dtype=np.float64)
        else:
            prod_lik = np.ones((K, L, 20), dtype=np.float64)
            acc_scale = np.zeros((K, L), dtype=np.float64)
            for child in node.children:
                P_batch = model.batched_transition_matrices(
                    child.branch_length, rates
                )  # (K, 20, 20)
                c_lik = cond_lik[child.node_id]  # (K, L, 20)
                # msg[k, s, a] = sum_b P[k, a, b] * c_lik[k, s, b]
                msg = np.einsum("kab,ksb->ksa", P_batch, c_lik)
                prod_lik *= msg
                acc_scale += log_scales[child.node_id]

            max_val = np.max(prod_lik, axis=2)  # (K, L)
            safe_max = np.where(max_val > 1e-300, max_val, 1.0)
            prod_lik /= safe_max[:, :, None]
            step_scale = np.where(max_val > 1e-300, np.log(safe_max), -700.0)
            cond_lik[node.node_id] = prod_lik
            log_scales[node.node_id] = acc_scale + step_scale

    root_lik = cond_lik[tree.node_id]  # (K, L, 20)
    root_scale = log_scales[tree.node_id]  # (K, L)
    cat_site = np.einsum("a,ksa->ks", model.pi, root_lik)
    cat_site = np.maximum(cat_site, 1e-300)
    log_cat = np.log(cat_site) + root_scale  # (K, L)

    # Log-sum-exp across K equiprobable Gamma categories
    max_k = np.max(log_cat, axis=0)
    site_ll = max_k + np.log(np.mean(np.exp(log_cat - max_k[None, :]), axis=0))
    return float(np.sum(site_ll))


def optimize_branch_lengths_ml(
    tree: TreeNode,
    msa: MultipleSequenceAlignment,
    model: SubstitutionModel,
    alpha: float = 1.0,
    n_gamma: int = 4,
    n_rounds: int = 2,
    optimize_alpha: bool = False,
) -> Tuple[TreeNode, float, float]:
    """Optimize branch lengths (and optionally Gamma alpha) via coordinate Brent ascent."""
    opt_tree = tree.clone()
    opt_tree.assign_ids()
    cur_alpha = float(alpha)

    for _ in range(max(1, n_rounds)):
        nodes = [n for n in opt_tree.postorder() if n is not opt_tree]
        for node in nodes:
            def _neg_ll(blen: float) -> float:
                node.branch_length = float(blen)
                return -felsenstein_log_likelihood(
                    opt_tree, msa, model, alpha=cur_alpha, n_gamma=n_gamma
                )

            res = minimize_scalar(
                _neg_ll,
                bounds=(1e-4, 4.0),
                method="bounded",
                options={"xatol": 1e-3, "maxiter": 18},
            )
            node.branch_length = float(res.x)

        if optimize_alpha and n_gamma > 1:
            def _neg_ll_alpha(a_val: float) -> float:
                return -felsenstein_log_likelihood(
                    opt_tree, msa, model, alpha=float(a_val), n_gamma=n_gamma
                )

            res_a = minimize_scalar(
                _neg_ll_alpha,
                bounds=(0.15, 10.0),
                method="bounded",
                options={"xatol": 1e-2, "maxiter": 14},
            )
            cur_alpha = float(res_a.x)

    final_ll = felsenstein_log_likelihood(
        opt_tree, msa, model, alpha=cur_alpha, n_gamma=n_gamma
    )
    return opt_tree, cur_alpha, final_ll


def _collect_internal_edges(root: TreeNode) -> List[Tuple[TreeNode, int]]:
    """Return (parent, child_idx) pairs where child is an internal bifurcating node."""
    edges: List[Tuple[TreeNode, int]] = []
    for node in root.preorder():
        for idx, ch in enumerate(node.children):
            if not ch.is_leaf and len(ch.children) == 2 and len(node.children) == 2:
                edges.append((node, idx))
    return edges


def _apply_nni_swap(parent: TreeNode, child_idx: int, swap_grandchild_idx: int) -> None:
    """Perform an in-place NNI swap across edge (parent, parent.children[child_idx])."""
    u = parent.children[child_idx]
    sibling_idx = 1 - child_idx
    sibling = parent.children[sibling_idx]
    moving_gc = u.children[swap_grandchild_idx]
    u.children[swap_grandchild_idx] = sibling
    parent.children[sibling_idx] = moving_gc


def search_topology_nni(
    tree: TreeNode,
    msa: MultipleSequenceAlignment,
    model: SubstitutionModel,
    alpha: float = 1.0,
    n_gamma: int = 4,
    max_rounds: int = 2,
) -> Tuple[TreeNode, float]:
    """Perform Maximum-Likelihood Nearest-Neighbor Interchange (NNI) hill climbing."""
    best_tree = tree.clone()
    best_tree.assign_ids()
    best_ll = felsenstein_log_likelihood(best_tree, msa, model, alpha=alpha, n_gamma=n_gamma)

    for _ in range(max_rounds):
        improved = False
        edges = _collect_internal_edges(best_tree)
        for edge_idx in range(len(edges)):
            for gc_idx in (0, 1):
                cand = best_tree.clone()
                cand_edges = _collect_internal_edges(cand)
                if edge_idx >= len(cand_edges):
                    continue
                p_node, c_idx = cand_edges[edge_idx]
                _apply_nni_swap(p_node, c_idx, gc_idx)
                cand.assign_ids()
                cand_ll = felsenstein_log_likelihood(
                    cand, msa, model, alpha=alpha, n_gamma=n_gamma
                )
                if cand_ll > best_ll + 1e-4:
                    # Refine local branch lengths around accepted topology
                    cand, _, cand_ll = optimize_branch_lengths_ml(
                        cand, msa, model, alpha=alpha, n_gamma=n_gamma, n_rounds=1
                    )
                    if cand_ll > best_ll + 1e-4:
                        best_tree = cand
                        best_ll = cand_ll
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    return best_tree, best_ll


def compute_aic_bic(
    log_likelihood: float, tree: TreeNode, alignment_length: int
) -> Dict[str, float]:
    """Compute AIC and BIC model selection criteria for a bifurcating phylogenetic tree."""
    n_taxa = len(tree.leaves())
    n_params = max(1, 2 * n_taxa - 3) + 1  # branches + alpha
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = n_params * float(np.log(max(2, alignment_length))) - 2.0 * log_likelihood
    return {"n_params": float(n_params), "aic": aic, "bic": bic}


def _tree_bipartitions(root: TreeNode) -> Set[FrozenSet[FrozenSet[str]]]:
    """Extract all non-trivial unrooted bipartition splits {A, B} from a tree."""
    all_leaves = frozenset(lf.name for lf in root.leaves())
    splits: Set[FrozenSet[FrozenSet[str]]] = set()

    def _gather(node: TreeNode) -> FrozenSet[str]:
        if node.is_leaf:
            return frozenset((node.name,))
        clade: Set[str] = set()
        for ch in node.children:
            clade.update(_gather(ch))
        fclade = frozenset(clade)
        comp = all_leaves - fclade
        if len(fclade) > 1 and len(comp) > 1:
            splits.add(frozenset((fclade, comp)))
        return fclade

    _gather(root)
    return splits


def _rooted_clades(root: TreeNode) -> Set[FrozenSet[str]]:
    """Extract all non-trivial rooted clades (size >= 2 and < N_taxa)."""
    n_taxa = len(root.leaves())
    clades: Set[FrozenSet[str]] = set()

    def _gather(node: TreeNode) -> FrozenSet[str]:
        if node.is_leaf:
            return frozenset((node.name,))
        s: Set[str] = set()
        for ch in node.children:
            s.update(_gather(ch))
        fs = frozenset(s)
        if 1 < len(fs) < n_taxa:
            clades.add(fs)
        return fs

    _gather(root)
    return clades


def robinson_foulds_distance(tree_a: TreeNode, tree_b: TreeNode) -> Tuple[int, float]:
    """Compute unrooted Robinson-Foulds symmetric split distance and normalized RF in [0, 1]."""
    splits_a = _tree_bipartitions(tree_a)
    splits_b = _tree_bipartitions(tree_b)
    sym_diff = len(splits_a ^ splits_b)
    max_splits = max(1, len(splits_a) + len(splits_b))
    return sym_diff, float(sym_diff) / float(max_splits)


def _compute_ess(trace: Sequence[float]) -> float:
    """Estimate Effective Sample Size (ESS) of a scalar MCMC trace via autocorrelation."""
    x = np.asarray(trace, dtype=np.float64)
    n = len(x)
    if n < 4:
        return float(n)
    xc = x - np.mean(x)
    var = float(np.dot(xc, xc))
    if var < 1e-12:
        return float(n)
    tau = 1.0
    max_lag = min(n // 2, 50)
    for lag in range(1, max_lag):
        rho = float(np.dot(xc[:-lag], xc[lag:])) / var
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return max(1.0, float(n) / tau)


def run_bayesian_mcmc(
    initial_tree: TreeNode,
    msa: MultipleSequenceAlignment,
    model: SubstitutionModel,
    config: Optional[PhyloConfig] = None,
) -> McmcSummary:
    """Run Metropolis-Hastings Bayesian MCMC over branch lengths, alpha, and NNI topologies.

    Priors:
      - Independent Exponential(rate=branch_prior_rate) on each branch length.
      - Exponential(rate=1.0) on Gamma shape alpha.
      - Uniform prior over labeled topologies.
    Returns posterior summaries, ESS, clade posterior supports, and annotated MAP tree.
    """
    cfg = config or PhyloConfig()
    rng = np.random.default_rng(cfg.seed)

    cur_tree = initial_tree.clone()
    cur_tree.assign_ids()
    cur_alpha = float(cfg.gamma_alpha)

    def _log_prior(t: TreeNode, a: float) -> float:
        blens = [max(1e-6, n.branch_length) for n in t.postorder() if n is not t]
        lp_b = float(
            len(blens) * np.log(cfg.branch_prior_rate) - cfg.branch_prior_rate * sum(blens)
        )
        lp_a = -float(a)
        return lp_b + lp_a

    cur_ll = felsenstein_log_likelihood(
        cur_tree, msa, model, alpha=cur_alpha, n_gamma=cfg.n_gamma_categories
    )
    cur_post = cur_ll + _log_prior(cur_tree, cur_alpha)

    map_tree = cur_tree.clone()
    map_post = cur_post

    burnin = int(cfg.mcmc_steps * cfg.burnin_fraction)
    accepted = 0
    post_ll_samples: List[float] = []
    post_tl_samples: List[float] = []
    post_alpha_samples: List[float] = []
    trace: List[float] = []
    clade_counts: Dict[FrozenSet[str], int] = {}
    bipart_counts: Dict[FrozenSet[FrozenSet[str]], int] = {}
    n_post_samples = 0

    for step in range(cfg.mcmc_steps):
        cand_tree = cur_tree.clone()
        cand_alpha = cur_alpha
        log_hastings = 0.0

        move_roll = float(rng.random())
        internal_edges = _collect_internal_edges(cand_tree)

        if move_roll < 0.70:
            # Log-normal multiplier proposal on one random branch
            non_root = [n for n in cand_tree.postorder() if n is not cand_tree]
            target = non_root[int(rng.integers(0, len(non_root)))]
            sigma = 0.28
            u = float(rng.normal(0.0, sigma))
            old_b = max(1e-5, target.branch_length)
            new_b = float(np.clip(old_b * np.exp(u), 1e-5, 5.0))
            target.branch_length = new_b
            log_hastings = float(np.log(new_b / old_b))
        elif move_roll < 0.85 and cfg.n_gamma_categories > 1:
            # Gamma shape alpha proposal
            u = float(rng.normal(0.0, 0.22))
            old_a = cur_alpha
            cand_alpha = float(np.clip(old_a * np.exp(u), 0.1, 10.0))
            log_hastings = float(np.log(cand_alpha / old_a))
        elif internal_edges:
            # Symmetric NNI topology proposal
            p_node, c_idx = internal_edges[int(rng.integers(0, len(internal_edges)))]
            gc_idx = int(rng.integers(0, 2))
            _apply_nni_swap(p_node, c_idx, gc_idx)
            cand_tree.assign_ids()

        cand_ll = felsenstein_log_likelihood(
            cand_tree, msa, model, alpha=cand_alpha, n_gamma=cfg.n_gamma_categories
        )
        cand_post = cand_ll + _log_prior(cand_tree, cand_alpha)
        log_accept = (cand_post - cur_post) + log_hastings

        if np.log(max(1e-15, float(rng.random()))) < log_accept:
            cur_tree = cand_tree
            cur_alpha = cand_alpha
            cur_ll = cand_ll
            cur_post = cand_post
            accepted += 1

        trace.append(cur_post)
        if step >= burnin:
            if n_post_samples == 0 or cur_post > map_post:
                map_post = cur_post
                map_tree = cur_tree.clone()
            n_post_samples += 1
            post_ll_samples.append(cur_ll)
            post_tl_samples.append(cur_tree.total_branch_length())
            post_alpha_samples.append(cur_alpha)
            for clade in _rooted_clades(cur_tree):
                clade_counts[clade] = clade_counts.get(clade, 0) + 1
            for bp in _tree_bipartitions(cur_tree):
                bipart_counts[bp] = bipart_counts.get(bp, 0) + 1

    denom = float(max(1, n_post_samples))
    clade_posteriors = {c: cnt / denom for c, cnt in clade_counts.items()}
    bipart_posteriors = {bp: cnt / denom for bp, cnt in bipart_counts.items()}
    all_leaves = frozenset(lf.name for lf in map_tree.leaves())

    # Annotate MAP tree internal nodes with Pulley-invariant bipartition posterior support
    def _annotate(node: TreeNode) -> FrozenSet[str]:
        if node.is_leaf:
            return frozenset((node.name,))
        s: Set[str] = set()
        for ch in node.children:
            s.update(_annotate(ch))
        fs = frozenset(s)
        comp = all_leaves - fs
        if node is not map_tree and len(fs) > 1 and len(comp) > 1:
            bp = frozenset((fs, comp))
            node.support = bipart_posteriors.get(bp, clade_posteriors.get(fs, 0.0))
        return fs

    _annotate(map_tree)

    return McmcSummary(
        map_tree=map_tree,
        map_log_posterior=map_post,
        mean_log_likelihood=float(np.mean(post_ll_samples)) if post_ll_samples else cur_ll,
        mean_tree_length=(
            float(np.mean(post_tl_samples)) if post_tl_samples else cur_tree.total_branch_length()
        ),
        mean_alpha=float(np.mean(post_alpha_samples)) if post_alpha_samples else cur_alpha,
        ess_log_likelihood=_compute_ess(post_ll_samples),
        acceptance_rate=float(accepted) / float(max(1, cfg.mcmc_steps)),
        clade_posteriors=clade_posteriors,
        log_posterior_trace=trace,
    )


def reconstruct_ancestral_sequences(
    tree: TreeNode,
    msa: MultipleSequenceAlignment,
    model: SubstitutionModel,
) -> AncestralReconstructionResult:
    """Perform Marginal Empirical-Bayes and Joint Pupko Ancestral Sequence Reconstruction.

    Marginal ASR computes exact site-wise posterior probabilities P(X_u = a | MSA, T)
    at every internal node via upward-downward message passing. Joint ASR computes the
    globally optimal joint ancestral character assignment across all internal nodes.
    """
    tree.assign_ids()
    name_map = _align_msa_to_tree(msa, tree)
    L = msa.length
    nodes = tree.postorder()

    # Map parent relationships
    parent_of: Dict[int, Optional[TreeNode]] = {tree.node_id: None}
    for node in nodes:
        for ch in node.children:
            parent_of[ch.node_id] = node

    # 1. Upward pass (single mean-rate model for clean marginal/joint posteriors)
    up_lik: Dict[int, FloatArray] = {}
    trans_mats: Dict[int, FloatArray] = {}
    for node in nodes:
        if node is not tree:
            trans_mats[node.node_id] = model.transition_matrix(node.branch_length)
        if node.is_leaf:
            seq_idx = name_map[node.name]
            arr = np.zeros((L, 20), dtype=np.float64)
            for s in range(L):
                aa = int(seq_idx[s])
                if aa < 20:
                    arr[s, aa] = 1.0
                else:
                    arr[s, :] = 1.0
            up_lik[node.node_id] = arr
        else:
            prod = np.ones((L, 20), dtype=np.float64)
            for ch in node.children:
                P = trans_mats[ch.node_id]
                msg = up_lik[ch.node_id] @ P.T  # msg[s, a] = sum_b P[a, b] * up[ch, s, b]
                prod *= msg
            mx = np.max(prod, axis=1, keepdims=True)
            prod /= np.where(mx > 1e-300, mx, 1.0)
            up_lik[node.node_id] = prod

    # 2. Downward pass for marginal posteriors
    down_lik: Dict[int, FloatArray] = {
        tree.node_id: np.broadcast_to(model.pi[None, :], (L, 20)).copy()
    }
    for node in tree.preorder():
        if node.is_leaf:
            continue
        p_down = down_lik[node.node_id]
        for ch in node.children:
            # Message from parent + sibling(s) into child ch
            env = p_down.copy()
            for sib in node.children:
                if sib is ch:
                    continue
                P_sib = trans_mats[sib.node_id]
                env *= up_lik[sib.node_id] @ P_sib.T
            P_ch = trans_mats[ch.node_id]
            # down_ch[s, b] = sum_a env[s, a] * P_ch[a, b]
            d_ch = env @ P_ch
            mx = np.max(d_ch, axis=1, keepdims=True)
            d_ch /= np.where(mx > 1e-300, mx, 1.0)
            down_lik[ch.node_id] = d_ch

    # Label internal nodes deterministically
    internal_labels: Dict[int, str] = {}
    int_counter = 0
    for node in tree.preorder():
        if not node.is_leaf:
            internal_labels[node.node_id] = node.name if node.name else f"Node_{int_counter}"
            int_counter += 1

    marginal_seqs: Dict[str, str] = {}
    marginal_posts: Dict[str, FloatArray] = {}
    confidences: List[float] = []

    for node in tree.preorder():
        if node.is_leaf:
            continue
        lbl = internal_labels[node.node_id]
        raw_post = up_lik[node.node_id] * down_lik[node.node_id]
        denom = np.sum(raw_post, axis=1, keepdims=True)
        post = raw_post / np.maximum(1e-300, denom)
        marginal_posts[lbl] = post
        best_idx = np.argmax(post, axis=1)
        marginal_seqs[lbl] = "".join(CANONICAL_AA[int(a)] for a in best_idx)
        confidences.append(float(np.mean(np.max(post, axis=1))))

    # 3. Pupko Joint ASR (log-domain dynamic programming)
    # best_child_state[node_id][s, parent_state] -> optimal amino acid at node
    joint_best_state: Dict[int, IntArray] = {}
    joint_log_val: Dict[int, FloatArray] = {}

    for node in nodes:
        if node is tree:
            continue
        P = np.log(np.maximum(1e-300, trans_mats[node.node_id]))  # (20, 20)
        if node.is_leaf:
            seq_idx = name_map[node.name]
            val = np.empty((L, 20), dtype=np.float64)
            best_s = np.empty((L, 20), dtype=np.int64)
            for s in range(L):
                aa = int(seq_idx[s])
                if aa < 20:
                    val[s, :] = P[:, aa]
                    best_s[s, :] = aa
                else:
                    # Unknown/gap picks best transition from parent state
                    best_s[s, :] = np.argmax(P, axis=1)
                    val[s, :] = np.max(P, axis=1)
            joint_log_val[node.node_id] = val
            joint_best_state[node.node_id] = best_s
        else:
            sum_children = np.zeros((L, 20), dtype=np.float64)
            for ch in node.children:
                sum_children += joint_log_val[ch.node_id]
            # score[s, p, a] = P[p, a] + sum_children[s, a]
            scores = P[None, :, :] + sum_children[:, None, :]  # (L, 20, 20)
            joint_best_state[node.node_id] = np.argmax(scores, axis=2)
            joint_log_val[node.node_id] = np.max(scores, axis=2)

    # Root optimal assignment
    root_sum = np.log(np.maximum(1e-300, model.pi))[None, :] + sum(
        joint_log_val[ch.node_id] for ch in tree.children
    )
    assigned: Dict[int, IntArray] = {tree.node_id: np.argmax(root_sum, axis=1)}
    for node in tree.preorder():
        if node is tree or node.is_leaf:
            continue
        p_node = parent_of[node.node_id]
        assert p_node is not None
        p_states = assigned[p_node.node_id]
        assigned[node.node_id] = joint_best_state[node.node_id][np.arange(L), p_states]

    joint_seqs: Dict[str, str] = {}
    for node in tree.preorder():
        if not node.is_leaf:
            lbl = internal_labels[node.node_id]
            joint_seqs[lbl] = "".join(CANONICAL_AA[int(a)] for a in assigned[node.node_id])

    return AncestralReconstructionResult(
        marginal_sequences=marginal_seqs,
        marginal_posteriors=marginal_posts,
        joint_sequences=joint_seqs,
        mean_posterior_confidence=float(np.mean(confidences)) if confidences else 1.0,
    )


def infer_phylogeny_pipeline(
    msa: MultipleSequenceAlignment,
    initial_tree: TreeNode,
    config: Optional[PhyloConfig] = None,
    run_mcmc: bool = True,
) -> Tuple[TreeNode, float, float, Optional[McmcSummary], AncestralReconstructionResult]:
    """Run full ML branch/topology optimization, optional Bayesian MCMC, and ASR."""
    cfg = config or PhyloConfig()
    model = SubstitutionModel.from_name(cfg.model_name)

    ml_tree, opt_alpha, ml_ll = optimize_branch_lengths_ml(
        initial_tree,
        msa,
        model,
        alpha=cfg.gamma_alpha,
        n_gamma=cfg.n_gamma_categories,
        n_rounds=cfg.ml_rounds,
        optimize_alpha=True,
    )
    ml_tree, ml_ll = search_topology_nni(
        ml_tree,
        msa,
        model,
        alpha=opt_alpha,
        n_gamma=cfg.n_gamma_categories,
        max_rounds=cfg.nni_max_rounds,
    )

    mcmc_res: Optional[McmcSummary] = None
    if run_mcmc and cfg.mcmc_steps > 0:
        mcmc_cfg = PhyloConfig(
            model_name=cfg.model_name,
            n_gamma_categories=cfg.n_gamma_categories,
            gamma_alpha=opt_alpha,
            mcmc_steps=cfg.mcmc_steps,
            burnin_fraction=cfg.burnin_fraction,
            branch_prior_rate=cfg.branch_prior_rate,
            seed=cfg.seed,
        )
        mcmc_res = run_bayesian_mcmc(ml_tree, msa, model, mcmc_cfg)

    asr = reconstruct_ancestral_sequences(ml_tree, msa, model)
    return ml_tree, opt_alpha, ml_ll, mcmc_res, asr
