"""Pure 7-bit ASCII terminal visualizers for contact maps, MSAs, trees, Ramachandran, and GNM."""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from .types import FloatArray, MultipleSequenceAlignment, TreeNode


def _assert_ascii(text: str) -> str:
    """Guarantee rendered output contains strictly 7-bit ASCII characters (0..127)."""
    return text.encode("ascii", errors="replace").decode("ascii")


def render_contact_map_ascii(
    score_matrix: FloatArray,
    true_contacts: Optional[NDArray[np.bool_]] = None,
    top_k: int = 35,
    min_seq_sep: int = 6,
    max_display_size: int = 44,
) -> str:
    """Render an ASCII protein contact map comparing top-k predicted pairs vs true 3D contacts.

    Legend:
      '#' = True Positive (predicted and in 3D contact)
      '+' = Predicted Only (top-k prediction outside 3D cutoff)
      '.' = True Contact Only (3D contact not in top-k)
      '/' = Sequence diagonal (|i - j| < min_seq_sep)
    """
    L = score_matrix.shape[0]
    step = max(1, (L + max_display_size - 1) // max_display_size)
    bins = (L + step - 1) // step

    pairs: List[tuple[int, int, float]] = []
    for i in range(L):
        for j in range(i + min_seq_sep, L):
            pairs.append((i, j, float(score_matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pred_mask = np.zeros((L, L), dtype=bool)
    for i, j, _ in pairs[:top_k]:
        pred_mask[i, j] = True
        pred_mask[j, i] = True

    lines: List[str] = [
        f"Contact Map (L={L}, step={step}, top_k={top_k}) "
        "[#=TP, +=Pred, .=True, /=Diag]"
    ]
    header = "     " + "".join(str((b * step) // 10 % 10) for b in range(bins))
    header2 = "     " + "".join(str((b * step) % 10) for b in range(bins))
    lines.append(header)
    lines.append(header2)
    lines.append("    +" + "-" * bins + "+")

    for bi in range(bins):
        i0, i1 = bi * step, min(L, (bi + 1) * step)
        row_chars: List[str] = []
        for bj in range(bins):
            j0, j1 = bj * step, min(L, (bj + 1) * step)
            if abs(bi - bj) * step < min_seq_sep and bi == bj:
                row_chars.append("/")
                continue
            p_hit = bool(np.any(pred_mask[i0:i1, j0:j1]))
            t_hit = (
                bool(np.any(true_contacts[i0:i1, j0:j1]))
                if true_contacts is not None
                else False
            )
            if p_hit and t_hit:
                row_chars.append("#")
            elif p_hit:
                row_chars.append("+")
            elif t_hit:
                row_chars.append(".")
            else:
                row_chars.append(" ")
        lines.append(f"{i0:3d} |{''.join(row_chars)}|")

    lines.append("    +" + "-" * bins + "+")
    return _assert_ascii("\n".join(lines))


def render_msa_alignment_ascii(
    msa: MultipleSequenceAlignment,
    conservation: Optional[FloatArray] = None,
    dssp_codes: Optional[str] = None,
    block_width: int = 60,
    max_seqs: int = 12,
) -> str:
    """Render a wrapped 7-bit ASCII MSA block with conservation symbols and optional DSSP track."""
    L = msa.length
    n_show = min(msa.n_seqs, max_seqs)
    name_w = min(16, max(len(n) for n in msa.names[:n_show]))
    consensus = msa.consensus_sequence()

    cons_track: Optional[str] = None
    if conservation is not None:
        chars: List[str] = []
        for c in conservation[:L]:
            if c >= 0.85:
                chars.append("*")
            elif c >= 0.60:
                chars.append(":")
            elif c >= 0.35:
                chars.append(".")
            else:
                chars.append(" ")
        cons_track = "".join(chars)

    lines: List[str] = [f"MSA View ({msa.n_seqs} seqs, {L} cols; showing {n_show})"]
    for start in range(0, L, block_width):
        end = min(L, start + block_width)
        lines.append(f"[{start + 1}..{end}]")
        for idx in range(n_show):
            nm = msa.names[idx][:name_w].ljust(name_w)
            lines.append(f"  {nm}  {msa.sequences[idx][start:end]}")
        lines.append(f"  {'Consensus'.ljust(name_w)}  {consensus[start:end]}")
        if cons_track is not None:
            lines.append(f"  {'Conservation'.ljust(name_w)}  {cons_track[start:end]}")
        if dssp_codes is not None and len(dssp_codes) >= end:
            lines.append(f"  {'DSSP_SS'.ljust(name_w)}  {dssp_codes[start:end]}")
        lines.append("")

    return _assert_ascii("\n".join(lines))


def render_tree_ascii(root: TreeNode) -> str:
    """Render a phylogenetic TreeNode hierarchy as a pure 7-bit ASCII cladogram."""
    lines: List[str] = ["Phylogenetic Tree (branch_length | support):"]

    def _walk(node: TreeNode, prefix: str, is_last: bool, is_root: bool) -> None:
        connector = "" if is_root else ("`-- " if is_last else "|-- ")
        if node.is_leaf:
            label = f"{node.name} (t={node.branch_length:.4f})"
        else:
            sup = f" [pp={node.support:.2f}]" if node.support is not None else ""
            lbl = node.name if node.name else "clade"
            if is_root:
                label = f"{lbl}{sup}"
            else:
                label = f"{lbl} (t={node.branch_length:.4f}){sup}"
        lines.append(f"{prefix}{connector}{label}")
        child_prefix = prefix if is_root else (prefix + ("    " if is_last else "|   "))
        for idx, ch in enumerate(node.children):
            _walk(ch, child_prefix, idx == len(node.children) - 1, False)

    _walk(root, "", True, True)
    return _assert_ascii("\n".join(lines))


def render_ramachandran_ascii(
    phi_deg: FloatArray,
    psi_deg: FloatArray,
    width: int = 36,
    height: int = 18,
) -> str:
    """Render a 2D ASCII Ramachandran (phi, psi) occupancy histogram over [-180, 180]^2."""
    grid = np.zeros((height, width), dtype=np.int64)
    for ph, ps in zip(phi_deg, psi_deg):
        if np.isnan(ph) or np.isnan(ps):
            continue
        col = int(np.clip(((ph + 180.0) / 360.0) * width, 0, width - 1))
        row = int(np.clip(((180.0 - ps) / 360.0) * height, 0, height - 1))
        grid[row, col] += 1

    lines: List[str] = ["Ramachandran Plot (phi: -180..+180, psi: +180..-180)"]
    lines.append("    +" + "-" * width + "+")
    mid_r = height // 2
    mid_c = width // 2

    for r in range(height):
        row_chars: List[str] = []
        for c in range(width):
            cnt = int(grid[r, c])
            if cnt >= 4:
                row_chars.append("#")
            elif cnt >= 2:
                row_chars.append("*")
            elif cnt == 1:
                row_chars.append("o")
            elif r == mid_r and c == mid_c:
                row_chars.append("+")
            elif r == mid_r:
                row_chars.append("-")
            elif c == mid_c:
                row_chars.append("|")
            else:
                row_chars.append(" ")
        psi_lbl = f"{int(180 - (r + 0.5) * 360.0 / height):4d}"
        lines.append(f"{psi_lbl}|" + "".join(row_chars) + "|")
    lines.append("    +" + "-" * width + "+")
    lines.append("    -180              0               +180")
    return _assert_ascii("\n".join(lines))


def render_dynamics_profile_ascii(
    bfactors: FloatArray,
    hinge_residues: Sequence[int],
    height: int = 8,
    max_cols: int = 64,
) -> str:
    """Render a 7-bit ASCII GNM B-factor flexibility profile with '^' hinge markers."""
    L = len(bfactors)
    step = max(1, (L + max_cols - 1) // max_cols)
    binned_b: List[float] = []
    binned_hinge: List[bool] = []
    hinge_set = set(hinge_residues)

    for start in range(0, L, step):
        end = min(L, start + step)
        binned_b.append(float(np.mean(bfactors[start:end])))
        binned_hinge.append(any(idx in hinge_set for idx in range(start, end)))

    vals = np.array(binned_b, dtype=np.float64)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = max(1e-9, hi - lo)
    norm_h = ((vals - lo) / span) * float(height - 1)

    lines: List[str] = [
        f"GNM Flexibility Profile (L={L}, min={lo:.2f}, max={hi:.2f}, '^'=hinge)"
    ]
    for level in range(height - 1, -1, -1):
        row = ["#" if nh >= level else " " for nh in norm_h]
        lines.append(f"{lo + span * (level / max(1, height - 1)):5.1f} |{''.join(row)}|")
    lines.append("      +" + "-" * len(binned_b) + "+")
    hinge_line = "".join("^" if h else " " for h in binned_hinge)
    lines.append(f"Hinge  {hinge_line}")
    return _assert_ascii("\n".join(lines))
