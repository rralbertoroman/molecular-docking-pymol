"""PyMOL automation helpers for the protein-protein docking workflow.

Each function reinitialises the PyMOL session so that calls are idempotent and
can be re-run from a notebook without leaking state between cells.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd
from pymol import cmd


PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


def _abs(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def fetch_or_load(pdb_or_path: str | Path, out_dir: str | Path) -> Path:
    """Return a path to a local .pdb for `pdb_or_path`.

    - 4-char string matching PDB_ID_RE → cmd.fetch into out_dir.
    - existing path → copied into out_dir to keep all inputs together.
    """
    out_dir = _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = str(pdb_or_path)

    if PDB_ID_RE.match(s):
        cmd.reinitialize()
        cmd.set("fetch_path", str(out_dir))
        cmd.fetch(s.lower(), type="pdb", async_=0)
        out_path = out_dir / f"{s.lower()}.pdb"
        if not out_path.exists():
            cif_path = out_dir / f"{s.lower()}.cif"
            if cif_path.exists():
                cmd.load(str(cif_path), "tmp")
                cmd.save(str(out_path), "tmp")
                cmd.delete("tmp")
        if not out_path.exists():
            raise FileNotFoundError(f"PyMOL fetch did not produce {out_path}")
        return out_path

    src = _abs(s)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = out_dir / src.name
    if src != dst:
        shutil.copy2(src, dst)
    return dst


def clean_structure(
    pdb_path: str | Path,
    out_path: str | Path,
    keep_chains: list[str] | None = None,
) -> Path:
    """Strip waters, inorganic ions, and alternate conformers (keep alt A)."""
    pdb_path = _abs(pdb_path)
    out_path = _abs(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")
    cmd.remove("resn HOH")
    cmd.remove("inorganic")
    cmd.remove("not alt ''+A")
    cmd.alter("all", "alt=''")
    if keep_chains:
        chain_sel = " or ".join(f"chain {c}" for c in keep_chains)
        cmd.remove(f"not ({chain_sel})")
    cmd.save(str(out_path), "mol")
    cmd.delete("mol")
    return out_path


def _setup_view(width: int, height: int) -> None:
    cmd.bg_color("white")
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.3)
    cmd.set("ray_opaque_background", 1)
    cmd.viewport(width, height)


def render_cartoon(
    pdb_path: str | Path,
    out_png: str | Path,
    color_by: Literal["chain", "ss", "element"] = "chain",
    highlight_residues: list[tuple[str, int]] | None = None,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Render a cartoon view to a ray-traced PNG with a white background."""
    pdb_path = _abs(pdb_path)
    out_png = _abs(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")
    cmd.hide("everything")
    cmd.show("cartoon")

    if color_by == "chain":
        cmd.util.cbc("mol")
    elif color_by == "ss":
        cmd.color("red", "ss H")
        cmd.color("yellow", "ss S")
        cmd.color("green", "ss L+''")
    elif color_by == "element":
        cmd.util.cbag("mol")

    if highlight_residues:
        selectors = [f"(chain {c} and resi {r})" for c, r in highlight_residues]
        sel = " or ".join(selectors)
        cmd.select("hl", sel)
        cmd.show("sticks", "hl")
        cmd.util.cnc("hl")
        cmd.delete("hl")

    _setup_view(width, height)
    cmd.orient("mol")
    cmd.zoom("mol", buffer=2.0)
    cmd.ray(width, height)
    cmd.png(str(out_png), dpi=150)
    cmd.delete("mol")
    return out_png


def render_surface(
    pdb_path: str | Path,
    out_png: str | Path,
    ligand_selection: str | None = None,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Render a surface of the structure; if a ligand selection is given, show it as sticks."""
    pdb_path = _abs(pdb_path)
    out_png = _abs(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")
    cmd.hide("everything")

    if ligand_selection:
        cmd.select("lig", f"mol and ({ligand_selection})")
        cmd.select("rec", "mol and not lig")
        cmd.show("surface", "rec")
        cmd.color("grey80", "rec")
        cmd.show("sticks", "lig")
        cmd.util.cnc("lig")
    else:
        cmd.show("surface", "mol")
        cmd.color("grey80", "mol")

    _setup_view(width, height)
    cmd.orient("mol")
    cmd.zoom("mol", buffer=2.0)
    cmd.ray(width, height)
    cmd.png(str(out_png), dpi=150)
    cmd.delete("all")
    return out_png


def interface_residues(
    complex_pdb: str | Path,
    chain_a: str,
    chain_b: str,
    cutoff: float = 5.0,
) -> pd.DataFrame:
    """Return residues from each chain that are within `cutoff` Å of the other chain."""
    complex_pdb = _abs(complex_pdb)

    cmd.reinitialize()
    cmd.load(str(complex_pdb), "cx")

    rows: list[dict] = []
    for side, this_chain, other_chain in (("A", chain_a, chain_b), ("B", chain_b, chain_a)):
        sel = f"iface_{side}"
        cmd.select(
            sel,
            f"byres (cx and chain {this_chain}) within {cutoff} of (cx and chain {other_chain})",
        )
        seen: set[tuple[str, int]] = set()
        cmd.iterate(
            f"{sel} and name CA",
            "seen.add((chain, int(resi))); rows.append({'side': side, 'chain': chain, 'resi': int(resi), 'resn': resn})",
            space={"rows": rows, "seen": seen, "side": side},
        )
        cmd.delete(sel)

    cmd.delete("cx")
    df = pd.DataFrame(rows, columns=["side", "chain", "resi", "resn"])
    return df.sort_values(["side", "resi"]).reset_index(drop=True)


def align_to_native(
    model_pdb: str | Path,
    native_pdb: str | Path,
    out_png: str | Path | None = None,
    method: Literal["align", "cealign", "super"] = "align",
    width: int = 1200,
    height: int = 900,
) -> dict:
    """Superpose `model_pdb` onto `native_pdb` and return RMSD info."""
    model_pdb = _abs(model_pdb)
    native_pdb = _abs(native_pdb)

    cmd.reinitialize()
    cmd.load(str(model_pdb), "model")
    cmd.load(str(native_pdb), "native")

    if method == "cealign":
        result = cmd.cealign("native", "model")
        info = {"rmsd": float(result["RMSD"]), "n_atoms": int(result["alignment_length"])}
    elif method == "super":
        r = cmd.super("model", "native")
        info = {"rmsd": float(r[0]), "n_atoms": int(r[1])}
    else:
        r = cmd.align("model", "native")
        info = {"rmsd": float(r[0]), "n_atoms": int(r[1])}

    if out_png is not None:
        out_png = _abs(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        cmd.hide("everything")
        cmd.show("cartoon")
        cmd.color("cyan", "model")
        cmd.color("magenta", "native")
        _setup_view(width, height)
        cmd.orient()
        cmd.zoom("all", buffer=2.0)
        cmd.ray(width, height)
        cmd.png(str(out_png), dpi=150)
        info["png"] = str(out_png)

    cmd.delete("all")
    return info


def save_interface_csv(df: pd.DataFrame, out_csv: str | Path) -> Path:
    out_csv = _abs(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
