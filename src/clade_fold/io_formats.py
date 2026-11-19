"""Pure-Python parsers and serializers for FASTA, A3M, PDB, Newick, and JSON telemetry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .types import (
    FloatArray,
    MultipleSequenceAlignment,
    ONE_TO_THREE,
    ProteinStructure,
    ResidueRecord,
    THREE_TO_ONE,
    TreeNode,
)


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    """Parse FASTA-formatted protein sequences into (header_id, sequence) tuples."""
    records: List[Tuple[str, str]] = []
    current_name: Optional[str] = None
    chunks: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith(">"):
            if current_name is not None:
                records.append((current_name, "".join(chunks)))
            header = line[1:].strip()
            current_name = header.split()[0] if header else f"seq_{len(records) + 1}"
            chunks = []
        else:
            chunks.append(line.replace(" ", ""))

    if current_name is not None:
        records.append((current_name, "".join(chunks)))

    if not records:
        raise ValueError("No valid FASTA records found")
    return records


def parse_a3m(text: str, strip_lowercase_inserts: bool = True) -> MultipleSequenceAlignment:
    """Parse A3M or aligned FASTA text into a MultipleSequenceAlignment.

    In A3M format, lowercase characters denote insertion states relative to the
    query profile. Stripping lowercase characters recovers the aligned match-state
    columns of uniform width.
    """
    raw_records = parse_fasta(text)
    names: List[str] = []
    seqs: List[str] = []
    for name, seq in raw_records:
        names.append(name)
        if strip_lowercase_inserts:
            cleaned = "".join(ch for ch in seq if not ch.islower())
        else:
            cleaned = seq.upper()
        seqs.append(cleaned)
    return MultipleSequenceAlignment(names=names, sequences=seqs)


def write_fasta(records: Sequence[Tuple[str, str]], line_width: int = 80) -> str:
    """Serialize (name, sequence) tuples to standard FASTA text."""
    lines: List[str] = []
    for name, seq in records:
        lines.append(f">{name}")
        for start in range(0, len(seq), line_width):
            lines.append(seq[start:start + line_width])
    return "\n".join(lines) + "\n"


def compute_virtual_cb(n: FloatArray, ca: FloatArray, c: FloatArray) -> FloatArray:
    """Construct ideal tetrahedral virtual C-beta coordinates from backbone (N, CA, C).

    Uses the standard L-amino acid tetrahedral projection with bond length
    d(CA-CB) = 1.522 Angstrom relative to the N-CA and C-CA bond frame.
    """
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    norm_c = float(np.linalg.norm(c_vec))
    if norm_a < 1e-8 or norm_b < 1e-8 or norm_c < 1e-8:
        return ca.copy()
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca
    # Normalize CA->CB vector to exact 1.522 Angstrom
    delta = cb - ca
    d_norm = float(np.linalg.norm(delta))
    if d_norm > 1e-8:
        cb = ca + (1.522 / d_norm) * delta
    return cb.astype(np.float64)


def parse_pdb(
    text: str, chain_id: Optional[str] = None, name: str = "structure"
) -> ProteinStructure:
    """Parse PDB ATOM records into a validated ProteinStructure.

    Selects the first encountered chain unless ``chain_id`` is explicitly provided,
    prefers altLoc ' ' or 'A', and builds tetrahedral virtual CB coordinates for
    Glycine or residues lacking an explicit CB atom.
    """
    residue_atoms: Dict[Tuple[str, int, str], Dict[str, Tuple[FloatArray, float]]] = {}
    residue_order: List[Tuple[str, int, str]] = []
    residue_names: Dict[Tuple[str, int, str], str] = {}
    target_chain = chain_id

    for line in text.splitlines():
        if line.startswith("ENDMDL") and residue_order:
            break
        if not line.startswith("ATOM  "):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        alt_loc = line[16]
        if alt_loc not in (" ", "A"):
            continue
        res_name = line[17:20].strip()
        ch = line[21].strip() or "A"
        if target_chain is None:
            target_chain = ch
        if ch != target_chain:
            continue
        try:
            resseq = int(line[22:26].strip())
        except ValueError:
            continue
        icode = line[26].strip()
        key = (ch, resseq, icode)

        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            continue
        bfac = 20.0
        if len(line) >= 66:
            try:
                bfac = float(line[60:66].strip())
            except ValueError:
                bfac = 20.0

        if key not in residue_atoms:
            residue_atoms[key] = {}
            residue_order.append(key)
            residue_names[key] = THREE_TO_ONE.get(res_name, "X")

        if atom_name in ("N", "CA", "C", "O", "CB") and atom_name not in residue_atoms[key]:
            residue_atoms[key][atom_name] = (np.array([x, y, z], dtype=np.float64), bfac)

    residues: List[ResidueRecord] = []
    for idx, key in enumerate(residue_order):
        atoms = residue_atoms[key]
        if not all(k in atoms for k in ("N", "CA", "C", "O")):
            continue
        n_xyz, _ = atoms["N"]
        ca_xyz, ca_bfac = atoms["CA"]
        c_xyz, _ = atoms["C"]
        o_xyz, _ = atoms["O"]
        if "CB" in atoms:
            cb_xyz, _ = atoms["CB"]
        else:
            cb_xyz = compute_virtual_cb(n_xyz, ca_xyz, c_xyz)
        residues.append(
            ResidueRecord(
                index=len(residues),
                resseq=key[1],
                chain_id=key[0],
                aa=residue_names[key],
                n=n_xyz,
                ca=ca_xyz,
                c=c_xyz,
                o=o_xyz,
                cb=cb_xyz,
                b_factor=ca_bfac,
            )
        )

    if not residues:
        raise ValueError("No complete backbone residues (N, CA, C, O) parsed from PDB text")
    return ProteinStructure(name=name, residues=residues, chain_id=target_chain or "A")


def write_pdb(structure: ProteinStructure) -> str:
    """Serialize a ProteinStructure (N, CA, C, O, CB) to standard PDB ATOM format."""
    lines: List[str] = [f"HEADER    {structure.name[:40]:<40s}"]
    serial = 1
    for res in structure.residues:
        res3 = ONE_TO_THREE.get(res.aa, "UNK")
        atom_list = [("N", res.n), ("CA", res.ca), ("C", res.c), ("O", res.o)]
        if res.aa != "G":
            atom_list.append(("CB", res.cb))
        for atom_name, xyz in atom_list:
            elem = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d}  {atom_name:<3s} {res3:>3s} {res.chain_id:1s}"
                f"{res.resseq:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"  1.00{res.b_factor:6.2f}          {elem:>2s}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def parse_newick(text: str) -> TreeNode:
    """Parse a Newick phylogenetic tree string into a TreeNode hierarchy.

    Supports arbitrary branching, quoted or unquoted labels, numeric internal
    support values (e.g. bootstrap/posterior), and scientific-notation branch lengths.
    """
    src = text.strip()
    if not src:
        raise ValueError("Newick string is empty")
    if src.endswith(";"):
        src = src[:-1].strip()

    pos = 0
    n = len(src)

    def skip_ws() -> None:
        nonlocal pos
        while pos < n and src[pos].isspace():
            pos += 1

    def parse_label() -> str:
        nonlocal pos
        skip_ws()
        if pos >= n:
            return ""
        if src[pos] in ("'", '"'):
            quote = src[pos]
            pos += 1
            start = pos
            while pos < n and src[pos] != quote:
                pos += 1
            if pos >= n:
                raise ValueError("Unterminated quoted taxon label in Newick")
            label = src[start:pos]
            pos += 1
            return label
        start = pos
        while pos < n and src[pos] not in "():,;":
            pos += 1
        return src[start:pos].strip()

    def parse_branch_length() -> float:
        nonlocal pos
        skip_ws()
        if pos < n and src[pos] == ":":
            pos += 1
            skip_ws()
            start = pos
            while pos < n and src[pos] not in "(),;":
                pos += 1
            val_str = src[start:pos].strip()
            try:
                return max(0.0, float(val_str))
            except ValueError as exc:
                raise ValueError(f"Invalid Newick branch length: {val_str!r}") from exc
        return 0.0

    def parse_subtree() -> TreeNode:
        nonlocal pos
        skip_ws()
        children: List[TreeNode] = []
        if pos < n and src[pos] == "(":
            pos += 1
            while True:
                children.append(parse_subtree())
                skip_ws()
                if pos < n and src[pos] == ",":
                    pos += 1
                    continue
                if pos < n and src[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"Expected ',' or ')' at Newick position {pos}")
        label = parse_label()
        blen = parse_branch_length()
        support: Optional[float] = None
        name = label
        if children and label:
            try:
                support = float(label)
                name = ""
            except ValueError:
                name = label
        return TreeNode(name=name, branch_length=blen, support=support, children=children)

    root = parse_subtree()
    skip_ws()
    if pos != n:
        raise ValueError(f"Unexpected trailing characters in Newick at pos {pos}: {src[pos:]!r}")
    root.assign_ids()
    return root


def write_newick(root: TreeNode, include_support: bool = True, precision: int = 6) -> str:
    """Serialize a TreeNode hierarchy to canonical Newick format."""

    def _fmt(node: TreeNode, is_root: bool) -> str:
        if node.is_leaf:
            label = node.name
        else:
            inner = ",".join(_fmt(ch, False) for ch in node.children)
            if include_support and node.support is not None:
                lbl = f"{node.support:.3f}"
            else:
                lbl = node.name
            label = f"({inner}){lbl}"
        if is_root:
            return label
        return f"{label}:{node.branch_length:.{precision}f}"

    return _fmt(root, True) + ";"


def compute_msa_digest(msa: MultipleSequenceAlignment) -> str:
    """Compute a deterministic SHA-256 provenance digest for an MSA."""
    hasher = hashlib.sha256()
    for name, seq in zip(msa.names, msa.sequences):
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(seq.encode("ascii", errors="replace"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def read_text_file(path: str | Path) -> str:
    """Read UTF-8 text file safely."""
    return Path(path).read_text(encoding="utf-8")


def write_json_report(data: Dict[str, Any], path: str | Path) -> None:
    """Write a JSON telemetry dictionary with deterministic formatting."""
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
