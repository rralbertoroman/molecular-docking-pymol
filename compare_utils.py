"""Biochemical comparison helpers for a set of protein structures.

Compares an arbitrary set of structures (PDB IDs or local ``.pdb`` files) all
against each other across four dimensions:

1. Sequence & mutations (pairwise alignment, % identity, mutation table).
2. Active-site conservation (catalytic motif residues + 3D geometry).
3. Structural superposition (global RMSD + per-residue Calpha deviation).
4. Physicochemical properties & secondary structure.

Like :mod:`pymol_utils`, every PyMOL function reinitialises the session so calls
are idempotent and safe to re-run from a notebook. Tabular results are returned
as pandas DataFrames. Biopython is imported lazily so the module loads even when
it is not installed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pymol import cmd

from pymol_utils import (
    _abs,
    _setup_view,
    align_to_native,
    clean_structure,
    fetch_or_load,
)


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

# PBP-6 (3IT9 numbering) catalytic machinery: SXXK (Ser40-Lys43), SXN (Ser106),
# KTG (Lys209). Matches POCKET_RESIDUES in ligand_docking.ipynb and the active
# site default in md_utils.analyze_md.
PBP6_ACTIVE_SITE: list[dict] = [
    {"name": "SxxK-Ser", "resid": 40, "resn": "SER", "atom": "OG"},
    {"name": "SxxK-Lys", "resid": 43, "resn": "LYS", "atom": "NZ"},
    {"name": "SxN-Ser", "resid": 106, "resn": "SER", "atom": "OG"},
    {"name": "KTG-Lys", "resid": 209, "resn": "LYS", "atom": "NZ"},
]

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common modified / alternate names mapped to their parent
    "MSE": "M", "SEC": "U", "HSD": "H", "HSE": "H", "HSP": "H",
}

# charge at pH 7, polarity class, and approximate side-chain volume (A^3).
AA_PROPERTIES: dict[str, dict] = {
    "A": {"charge": 0, "polarity": "nonpolar", "size": 88.6},
    "R": {"charge": 1, "polarity": "charged", "size": 173.4},
    "N": {"charge": 0, "polarity": "polar", "size": 114.1},
    "D": {"charge": -1, "polarity": "charged", "size": 111.1},
    "C": {"charge": 0, "polarity": "polar", "size": 108.5},
    "Q": {"charge": 0, "polarity": "polar", "size": 143.8},
    "E": {"charge": -1, "polarity": "charged", "size": 138.4},
    "G": {"charge": 0, "polarity": "nonpolar", "size": 60.1},
    "H": {"charge": 0, "polarity": "polar", "size": 153.2},
    "I": {"charge": 0, "polarity": "nonpolar", "size": 166.7},
    "L": {"charge": 0, "polarity": "nonpolar", "size": 166.7},
    "K": {"charge": 1, "polarity": "charged", "size": 168.6},
    "M": {"charge": 0, "polarity": "nonpolar", "size": 162.9},
    "F": {"charge": 0, "polarity": "nonpolar", "size": 189.9},
    "P": {"charge": 0, "polarity": "nonpolar", "size": 112.7},
    "S": {"charge": 0, "polarity": "polar", "size": 89.0},
    "T": {"charge": 0, "polarity": "polar", "size": 116.1},
    "W": {"charge": 0, "polarity": "nonpolar", "size": 227.8},
    "Y": {"charge": 0, "polarity": "polar", "size": 193.6},
    "V": {"charge": 0, "polarity": "nonpolar", "size": 140.0},
}


# --------------------------------------------------------------------------- #
# Structure model + preparation
# --------------------------------------------------------------------------- #

@dataclass
class Structure:
    """One structure to compare: a label, a source, and a chain of interest."""

    id: str
    source: str | Path
    chain: str = "A"
    raw: Path | None = None
    clean: Path | None = None


def prepare_structures(
    specs: list[dict | Structure],
    out_dir: str | Path,
) -> list[Structure]:
    """Fetch/copy and clean each input, isolating its comparison chain.

    Each spec is a ``Structure`` or a dict with keys ``id``, ``source`` and an
    optional ``chain`` (default ``"A"``). Reuses ``fetch_or_load`` and
    ``clean_structure`` from pymol_utils.
    """
    out_dir = _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    structures: list[Structure] = []
    for spec in specs:
        s = spec if isinstance(spec, Structure) else Structure(**spec)
        s.raw = fetch_or_load(s.source, out_dir)
        clean_path = out_dir / f"{s.id}_clean.pdb"
        s.clean = clean_structure(s.raw, clean_path, keep_chains=[s.chain])
        structures.append(s)
    return structures


# --------------------------------------------------------------------------- #
# Dimension 1 — sequence & mutations
# --------------------------------------------------------------------------- #

def extract_sequence(pdb_path: str | Path, chain: str = "A") -> dict:
    """Return the polymer sequence of `chain` keyed to its real PDB numbering.

    Only residues bearing a CA are emitted, so unresolved loops naturally show
    up as alignment gaps downstream.
    """
    pdb_path = _abs(pdb_path)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")
    rows: list[dict] = []
    cmd.iterate(
        f"mol and chain {chain} and polymer and name CA",
        "rows.append({'resi': int(resi), 'resn': resn})",
        space={"rows": rows},
    )
    cmd.delete("mol")

    # de-dup on resi while preserving order (alt confs already stripped by clean)
    seen: set[int] = set()
    resids: list[int] = []
    resns: list[str] = []
    for r in rows:
        if r["resi"] in seen:
            continue
        seen.add(r["resi"])
        resids.append(r["resi"])
        resns.append(r["resn"])

    seq = "".join(THREE_TO_ONE.get(rn, "X") for rn in resns)
    return {"chain": chain, "seq": seq, "resids": resids, "resns": resns}


def align_sequences(
    seq_a: dict,
    seq_b: dict,
    mode: Literal["global", "local"] = "global",
) -> dict:
    """Pairwise-align two sequences (BLOSUM62) and report % identity.

    Identity is computed over columns where both sequences have a residue, so
    insertions/deletions do not distort the percentage.
    """
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Biopython is required for sequence alignment. Run `uv sync`."
        ) from exc

    aligner = PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    aln = aligner.align(seq_a["seq"], seq_b["seq"])[0]
    aligned_a, aligned_b = _aln_strings(aln)

    n_aligned = 0
    n_identical = 0
    for ca, cb in zip(aligned_a, aligned_b):
        if ca == "-" or cb == "-":
            continue
        n_aligned += 1
        if ca == cb:
            n_identical += 1

    identity_pct = 100.0 * n_identical / n_aligned if n_aligned else 0.0
    return {
        "identity_pct": identity_pct,
        "aligned_a": aligned_a,
        "aligned_b": aligned_b,
        "n_aligned": n_aligned,
        "n_identical": n_identical,
        "score": float(aln.score),
    }


def _aln_strings(aln) -> tuple[str, str]:
    """Extract the two gapped strings from a Biopython Alignment object."""
    # aln[0] / aln[1] yield the gapped sequences as strings.
    return str(aln[0]), str(aln[1])


def mutation_table(
    seq_a: dict,
    seq_b: dict,
    mode: Literal["global", "local"] = "global",
) -> pd.DataFrame:
    """Residue-by-residue substitution / insertion / deletion table.

    Alignment columns are mapped back to each structure's real PDB numbering so
    the table is meaningful even when the two structures number differently.
    """
    aln = align_sequences(seq_a, seq_b, mode=mode)
    aligned_a, aligned_b = aln["aligned_a"], aln["aligned_b"]

    ia = ib = 0
    rows: list[dict] = []
    for ca, cb in zip(aligned_a, aligned_b):
        resid_a = seq_a["resids"][ia] if ca != "-" else None
        resn_a = seq_a["resns"][ia] if ca != "-" else None
        resid_b = seq_b["resids"][ib] if cb != "-" else None
        resn_b = seq_b["resns"][ib] if cb != "-" else None

        if ca == "-":
            kind = "insertion"          # present in B, absent in A
        elif cb == "-":
            kind = "deletion"           # present in A, absent in B
        elif ca == cb:
            kind = "identity"
        else:
            kind = "substitution"

        rows.append({
            "resid_a": resid_a, "resn_a": resn_a, "aa_a": None if ca == "-" else ca,
            "resid_b": resid_b, "resn_b": resn_b, "aa_b": None if cb == "-" else cb,
            "kind": kind,
        })

        if ca != "-":
            ia += 1
        if cb != "-":
            ib += 1

    return pd.DataFrame(
        rows,
        columns=["resid_a", "resn_a", "aa_a", "resid_b", "resn_b", "aa_b", "kind"],
    )


# --------------------------------------------------------------------------- #
# Dimension 2 — active-site conservation
# --------------------------------------------------------------------------- #

def map_active_site(
    active_site: list[dict],
    seq_ref: dict,
    seq_target: dict,
) -> list[dict]:
    """Translate active-site residue numbers from `seq_ref` to `seq_target`.

    ``PBP6_ACTIVE_SITE`` is numbered against a reference structure (3IT9). A
    differently-numbered structure (e.g. a SwissModel that starts at residue 3)
    has the same catalytic residues at different residue numbers, so we map each
    reference resid through the pairwise alignment. Each returned site keeps a
    ``ref_resid`` and gets ``resid`` set to the aligned target number (``None``
    if the position is unaligned / missing in the target).
    """
    aln = align_sequences(seq_ref, seq_target)
    ia = ib = 0
    ref_to_target: dict[int, int] = {}
    for ca, cb in zip(aln["aligned_a"], aln["aligned_b"]):
        if ca != "-" and cb != "-":
            ref_to_target[seq_ref["resids"][ia]] = seq_target["resids"][ib]
        if ca != "-":
            ia += 1
        if cb != "-":
            ib += 1

    mapped: list[dict] = []
    for site in active_site:
        m = dict(site)
        m["ref_resid"] = site["resid"]
        m["resid"] = ref_to_target.get(site["resid"])
        mapped.append(m)
    return mapped


def active_site_residues(
    pdb_path: str | Path,
    chain: str = "A",
    active_site: list[dict] = PBP6_ACTIVE_SITE,
) -> pd.DataFrame:
    """Report which residue actually sits at each catalytic position.

    `active_site` must be numbered in this structure's own numbering (use
    :func:`map_active_site` first for a differently-numbered structure). A site
    with ``resid is None`` (unmapped) is reported as absent.
    """
    pdb_path = _abs(pdb_path)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")

    rows: list[dict] = []
    for site in active_site:
        observed = None
        if site["resid"] is not None:
            found: list[str] = []
            cmd.iterate(
                f"mol and chain {chain} and resi {site['resid']} and name CA",
                "found.append(resn)",
                space={"found": found},
            )
            observed = found[0] if found else None
        present = observed is not None
        conserved = present and observed == site["resn"]
        rows.append({
            "name": site["name"],
            "ref_resid": site.get("ref_resid", site["resid"]),
            "resid": site["resid"],
            "expected_resn": site["resn"],
            "observed_resn": observed,
            "present": present,
            "conserved": conserved,
        })
    cmd.delete("mol")
    return pd.DataFrame(
        rows,
        columns=["name", "ref_resid", "resid", "expected_resn",
                 "observed_resn", "present", "conserved"],
    )


def catalytic_geometry(
    pdb_path: str | Path,
    chain: str = "A",
    active_site: list[dict] = PBP6_ACTIVE_SITE,
) -> pd.DataFrame:
    """Pairwise distances between the catalytic atoms (3D geometry check).

    `active_site` must be in this structure's numbering (see
    :func:`map_active_site`). A pair's distance is ``None`` (NaN) when either
    atom is missing, rather than a misleading 0.
    """
    pdb_path = _abs(pdb_path)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")

    coords: dict[str, np.ndarray | None] = {}
    for site in active_site:
        xyz: list[tuple[float, float, float]] = []
        if site["resid"] is not None:
            cmd.iterate_state(
                1,
                f"mol and chain {chain} and resi {site['resid']} and name {site['atom']}",
                "xyz.append((x, y, z))",
                space={"xyz": xyz},
            )
        coords[site["name"]] = np.array(xyz[0]) if xyz else None
    cmd.delete("mol")

    rows: list[dict] = []
    for site_i, site_j in itertools.combinations(active_site, 2):
        ci, cj = coords[site_i["name"]], coords[site_j["name"]]
        dist = float(np.linalg.norm(ci - cj)) if ci is not None and cj is not None else None
        rows.append({
            "pair": f"{site_i['name']}--{site_j['name']}",
            "resid_i": site_i["resid"], "atom_i": site_i["atom"],
            "resid_j": site_j["resid"], "atom_j": site_j["atom"],
            "distance": dist,
        })
    return pd.DataFrame(
        rows,
        columns=["pair", "resid_i", "atom_i", "resid_j", "atom_j", "distance"],
    )


def compare_active_sites(
    struct_a: Structure,
    struct_b: Structure,
    active_site: list[dict] = PBP6_ACTIVE_SITE,
    ref_seq: dict | None = None,
) -> dict:
    """Side-by-side catalytic conservation + geometry drift for a pair.

    `active_site` is numbered against `ref_seq` (the sequence whose numbering the
    motif list uses; defaults to struct_a's sequence). The motif is mapped into
    each structure's own numbering before lookup, so it works even when the two
    structures number differently.
    """
    seq_a = extract_sequence(struct_a.clean, struct_a.chain)
    seq_b = extract_sequence(struct_b.clean, struct_b.chain)
    if ref_seq is None:
        ref_seq = seq_a

    site_a = map_active_site(active_site, ref_seq, seq_a)
    site_b = map_active_site(active_site, ref_seq, seq_b)

    res_a = active_site_residues(struct_a.clean, struct_a.chain, site_a)
    res_b = active_site_residues(struct_b.clean, struct_b.chain, site_b)
    res = res_a.merge(res_b, on=["name", "ref_resid", "expected_resn"], suffixes=("_a", "_b"))
    res["conserved_both"] = res["conserved_a"] & res["conserved_b"]

    geo_a = catalytic_geometry(struct_a.clean, struct_a.chain, site_a)
    geo_b = catalytic_geometry(struct_b.clean, struct_b.chain, site_b)
    geo = geo_a[["pair", "distance"]].merge(
        geo_b[["pair", "distance"]], on="pair", suffixes=("_a", "_b")
    )
    geo["ddist"] = (geo["distance_a"] - geo["distance_b"]).abs()

    return {"residues": res, "geometry": geo}


# --------------------------------------------------------------------------- #
# Dimension 3 — structural superposition
# --------------------------------------------------------------------------- #

def superpose_pair(
    struct_a: Structure,
    struct_b: Structure,
    out_png: str | Path | None = None,
    method: Literal["align", "cealign", "super"] = "cealign",
) -> dict:
    """Superpose B onto A and return RMSD info (reuses align_to_native)."""
    return align_to_native(struct_b.clean, struct_a.clean, out_png=out_png, method=method)


def per_residue_ca_deviation(
    struct_a: Structure,
    struct_b: Structure,
    method: Literal["align", "cealign", "super"] = "cealign",
) -> pd.DataFrame:
    """Per-residue Calpha deviation after superposition, by alignment column.

    Residues are matched by sequence-alignment column (not raw resid), so this
    survives differing numbering between crystal and model.
    """
    # 1) superpose in a PyMOL session and read back the transformed CA coords.
    cmd.reinitialize()
    cmd.load(str(_abs(struct_a.clean)), "native")   # reference (A)
    cmd.load(str(_abs(struct_b.clean)), "mobile")   # moved onto A (B)

    if method == "cealign":
        cmd.cealign("native", "mobile")
    elif method == "super":
        cmd.super("mobile", "native")
    else:
        cmd.align("mobile", "native")

    def _ca_coords(obj: str, chain: str) -> dict[int, np.ndarray]:
        rows: list[dict] = []
        cmd.iterate_state(
            1,
            f"{obj} and chain {chain} and polymer and name CA",
            "rows.append({'resi': int(resi), 'xyz': (x, y, z)})",
            space={"rows": rows},
        )
        out: dict[int, np.ndarray] = {}
        for r in rows:
            out.setdefault(r["resi"], np.array(r["xyz"]))
        return out

    ca_a = _ca_coords("native", struct_a.chain)
    ca_b = _ca_coords("mobile", struct_b.chain)
    cmd.delete("all")

    # 2) map alignment columns -> resids, then measure matched CA distances.
    seq_a = extract_sequence(struct_a.clean, struct_a.chain)
    seq_b = extract_sequence(struct_b.clean, struct_b.chain)
    aln = align_sequences(seq_a, seq_b)

    ia = ib = 0
    rows: list[dict] = []
    for ca, cb in zip(aln["aligned_a"], aln["aligned_b"]):
        if ca != "-" and cb != "-":
            resid_a = seq_a["resids"][ia]
            resid_b = seq_b["resids"][ib]
            pa, pb = ca_a.get(resid_a), ca_b.get(resid_b)
            dev = float(np.linalg.norm(pa - pb)) if pa is not None and pb is not None else None
            rows.append({
                "resid_a": resid_a, "resn_a": seq_a["resns"][ia],
                "resid_b": resid_b, "ca_dev_A": dev,
            })
        if ca != "-":
            ia += 1
        if cb != "-":
            ib += 1

    return pd.DataFrame(rows, columns=["resid_a", "resn_a", "resid_b", "ca_dev_A"])


# --------------------------------------------------------------------------- #
# Dimension 4 — physicochemical & secondary structure
# --------------------------------------------------------------------------- #

def secondary_structure(pdb_path: str | Path, chain: str = "A") -> pd.DataFrame:
    """Per-residue secondary structure (H/S/L) via PyMOL's dss assignment."""
    pdb_path = _abs(pdb_path)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")
    cmd.dss("mol")

    rows: list[dict] = []
    cmd.iterate(
        f"mol and chain {chain} and polymer and name CA",
        "rows.append({'resi': int(resi), 'resn': resn, 'ss': ss or 'L'})",
        space={"rows": rows},
    )
    cmd.delete("mol")

    seen: set[int] = set()
    clean_rows: list[dict] = []
    for r in rows:
        if r["resi"] in seen:
            continue
        seen.add(r["resi"])
        # PyMOL uses 'H' (helix), 'S' (strand), '' / 'L' (loop)
        ss = r["ss"] if r["ss"] in ("H", "S") else "L"
        clean_rows.append({"resid": r["resi"], "resn": r["resn"], "ss": ss})
    return pd.DataFrame(clean_rows, columns=["resid", "resn", "ss"])


def compare_secondary_structure(
    struct_a: Structure,
    struct_b: Structure,
) -> pd.DataFrame:
    """SS-change table joined on the sequence alignment columns."""
    ss_a = secondary_structure(struct_a.clean, struct_a.chain).set_index("resid")
    ss_b = secondary_structure(struct_b.clean, struct_b.chain).set_index("resid")

    seq_a = extract_sequence(struct_a.clean, struct_a.chain)
    seq_b = extract_sequence(struct_b.clean, struct_b.chain)
    aln = align_sequences(seq_a, seq_b)

    ia = ib = 0
    rows: list[dict] = []
    for ca, cb in zip(aln["aligned_a"], aln["aligned_b"]):
        if ca != "-" and cb != "-":
            resid_a = seq_a["resids"][ia]
            resid_b = seq_b["resids"][ib]
            sa = ss_a.loc[resid_a, "ss"] if resid_a in ss_a.index else None
            sb = ss_b.loc[resid_b, "ss"] if resid_b in ss_b.index else None
            rows.append({
                "resid_a": resid_a, "resn_a": seq_a["resns"][ia], "ss_a": sa,
                "resid_b": resid_b, "resn_b": seq_b["resns"][ib], "ss_b": sb,
                "ss_changed": sa != sb,
            })
        if ca != "-":
            ia += 1
        if cb != "-":
            ib += 1

    return pd.DataFrame(
        rows,
        columns=["resid_a", "resn_a", "ss_a", "resid_b", "resn_b", "ss_b", "ss_changed"],
    )


def classify_substitutions(mut_df: pd.DataFrame) -> pd.DataFrame:
    """Add charge/polarity/size changes + severity to substitution rows."""
    subs = mut_df[mut_df["kind"] == "substitution"].copy()

    def _prop(aa: str, key: str):
        return AA_PROPERTIES.get(aa, {}).get(key)

    rows: list[dict] = []
    for _, r in subs.iterrows():
        aa_a, aa_b = r["aa_a"], r["aa_b"]
        chg_a, chg_b = _prop(aa_a, "charge"), _prop(aa_b, "charge")
        pol_a, pol_b = _prop(aa_a, "polarity"), _prop(aa_b, "polarity")
        sz_a, sz_b = _prop(aa_a, "size"), _prop(aa_b, "size")

        charge_change = chg_a is not None and chg_b is not None and chg_a != chg_b
        polarity_change = pol_a is not None and pol_b is not None and pol_a != pol_b
        size_delta = (sz_b - sz_a) if sz_a is not None and sz_b is not None else None
        severity = int(bool(charge_change)) + int(bool(polarity_change)) + (
            1 if size_delta is not None and abs(size_delta) > 40 else 0
        )
        rows.append({
            "resid_a": r["resid_a"], "aa_a": aa_a,
            "resid_b": r["resid_b"], "aa_b": aa_b,
            "charge_a": chg_a, "charge_b": chg_b, "charge_change": charge_change,
            "polarity_a": pol_a, "polarity_b": pol_b, "polarity_change": polarity_change,
            "size_delta": size_delta, "severity": severity,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("severity", ascending=False).reset_index(drop=True)
    return df


def pocket_environment_summary(
    struct: Structure,
    active_site: list[dict] = PBP6_ACTIVE_SITE,
    radius: float = 8.0,
    ref_seq: dict | None = None,
) -> pd.DataFrame:
    """Residues within `radius` of any catalytic atom, with property classes.

    `active_site` is numbered against `ref_seq`; when `ref_seq` is given it is
    mapped into this structure's numbering first (so a renumbered model still
    finds its real pocket).
    """
    pdb_path = _abs(struct.clean)
    chain = struct.chain

    if ref_seq is not None:
        active_site = map_active_site(
            active_site, ref_seq, extract_sequence(pdb_path, chain)
        )
    active_site = [s for s in active_site if s["resid"] is not None]

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")

    atom_sels = [
        f"(chain {chain} and resi {s['resid']} and name {s['atom']})"
        for s in active_site
    ]
    cmd.select("catalytic", "mol and (" + " or ".join(atom_sels) + ")")
    cmd.select("pocket", f"byres (mol and polymer within {radius} of catalytic)")

    rows: list[dict] = []
    cmd.iterate(
        "pocket and name CA",
        "rows.append({'resi': int(resi), 'resn': resn})",
        space={"rows": rows},
    )
    cmd.delete("all")

    seen: set[int] = set()
    out: list[dict] = []
    for r in sorted(rows, key=lambda x: x["resi"]):
        if r["resi"] in seen:
            continue
        seen.add(r["resi"])
        aa = THREE_TO_ONE.get(r["resn"], "X")
        props = AA_PROPERTIES.get(aa, {})
        out.append({
            "resid": r["resi"], "resn": r["resn"], "aa": aa,
            "charge": props.get("charge"), "polarity": props.get("polarity"),
            "size": props.get("size"),
        })
    return pd.DataFrame(out, columns=["resid", "resn", "aa", "charge", "polarity", "size"])


# --------------------------------------------------------------------------- #
# All-vs-all matrices + orchestration
# --------------------------------------------------------------------------- #

def identity_matrix(
    structures: list[Structure],
    sequences: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Square matrix of pairwise % sequence identity (diagonal = 100)."""
    if sequences is None:
        sequences = {s.id: extract_sequence(s.clean, s.chain) for s in structures}

    ids = [s.id for s in structures]
    mat = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
    for s in structures:
        mat.loc[s.id, s.id] = 100.0
    for a, b in itertools.combinations(structures, 2):
        pct = align_sequences(sequences[a.id], sequences[b.id])["identity_pct"]
        mat.loc[a.id, b.id] = pct
        mat.loc[b.id, a.id] = pct
    return mat


def rmsd_matrix(
    structures: list[Structure],
    method: Literal["align", "cealign", "super"] = "cealign",
) -> pd.DataFrame:
    """Square matrix of pairwise superposition RMSD in Angstrom (diagonal = 0)."""
    ids = [s.id for s in structures]
    mat = pd.DataFrame(np.nan, index=ids, columns=ids, dtype=float)
    for s in structures:
        mat.loc[s.id, s.id] = 0.0
    for a, b in itertools.combinations(structures, 2):
        rmsd = superpose_pair(a, b, method=method)["rmsd"]
        mat.loc[a.id, b.id] = rmsd
        mat.loc[b.id, a.id] = rmsd
    return mat


def compare_all_pairs(
    structures: list[Structure],
    active_site: list[dict] = PBP6_ACTIVE_SITE,
    out_dir: str | Path | None = None,
    method: Literal["align", "cealign", "super"] = "cealign",
    active_site_ref: str | None = None,
) -> dict:
    """Run the full comparison over every unordered pair of structures.

    Returns ``{"identity_matrix", "rmsd_matrix", "pairs": {(id_a, id_b): {...}}}``.
    If `out_dir` is given, every matrix and per-pair table is written to CSV.
    `active_site` is numbered against `active_site_ref` (default: the first
    structure); the motif is mapped into each structure's numbering.
    """
    sequences = {s.id: extract_sequence(s.clean, s.chain) for s in structures}
    ref_id = active_site_ref or structures[0].id
    ref_seq = sequences[ref_id]
    id_mat = identity_matrix(structures, sequences)
    rms_mat = rmsd_matrix(structures, method=method)

    pairs: dict[tuple[str, str], dict] = {}
    for a, b in itertools.combinations(structures, 2):
        muts = mutation_table(sequences[a.id], sequences[b.id])
        site = compare_active_sites(a, b, active_site, ref_seq=ref_seq)
        result = {
            "alignment": align_sequences(sequences[a.id], sequences[b.id]),
            "mutations": muts,
            "active_site": site["residues"],
            "geometry": site["geometry"],
            "ca_deviation": per_residue_ca_deviation(a, b, method=method),
            "ss": compare_secondary_structure(a, b),
            "phys": classify_substitutions(muts),
        }
        pairs[(a.id, b.id)] = result

        if out_dir is not None:
            tag = f"{a.id}_vs_{b.id}"
            d = _abs(out_dir)
            save_csv(result["mutations"], d / f"mutations_{tag}.csv")
            save_csv(result["active_site"], d / f"active_site_{tag}.csv")
            save_csv(result["geometry"], d / f"catalytic_geometry_{tag}.csv")
            save_csv(result["ca_deviation"], d / f"ca_deviation_{tag}.csv")
            save_csv(result["ss"], d / f"ss_{tag}.csv")
            save_csv(result["phys"], d / f"substitutions_{tag}.csv")

    if out_dir is not None:
        d = _abs(out_dir)
        id_mat.to_csv(d / "identity_matrix.csv")
        rms_mat.to_csv(d / "rmsd_matrix.csv")

    return {"identity_matrix": id_mat, "rmsd_matrix": rms_mat, "pairs": pairs}


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def save_csv(df: pd.DataFrame, out_csv: str | Path) -> Path:
    out_csv = _abs(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def plot_ca_deviation(
    dev_df: pd.DataFrame,
    out_png: str | Path,
    active_site: list[dict] = PBP6_ACTIVE_SITE,
    title: str = "Desviacion Calpha por residuo",
) -> Path:
    """Plot per-residue CA deviation, marking catalytic residues."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = _abs(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = dev_df.dropna(subset=["ca_dev_A"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["resid_a"], df["ca_dev_A"], color="#2c3e50", linewidth=0.9)
    ax.fill_between(df["resid_a"], df["ca_dev_A"], color="#2c3e50", alpha=0.15)
    for site in active_site:
        ax.axvline(site["resid"], color="#c0392b", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Residuo (numeracion estructura A)")
    ax.set_ylabel(r"Desviacion C$\alpha$ ($\AA$)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_matrix_heatmap(
    matrix_df: pd.DataFrame,
    out_png: str | Path,
    title: str,
    cmap: str = "viridis",
) -> Path:
    """Render a square comparison matrix (identity or RMSD) as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = _abs(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(1.4 * len(matrix_df) + 2, 1.4 * len(matrix_df) + 1.5))
    im = ax.imshow(matrix_df.values, cmap=cmap)
    ax.set_xticks(range(len(matrix_df)))
    ax.set_yticks(range(len(matrix_df)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix_df.index)
    for i in range(len(matrix_df)):
        for j in range(len(matrix_df)):
            val = matrix_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
