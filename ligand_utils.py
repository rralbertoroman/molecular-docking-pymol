"""Ligand prep + AutoDock Vina docking helpers, layered on pymol_utils."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

from pymol import cmd

from pymol_utils import clean_structure


PUBCHEM_SDF_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
USER_AGENT = "molecular-docking-pymol/0.1"


def _abs(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _setup_view(width: int, height: int) -> None:
    cmd.bg_color("white")
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.3)
    cmd.set("ray_opaque_background", 1)
    cmd.viewport(width, height)


def _pubchem_fetch_sdf(cid: str, out_path: Path) -> Path:
    """Fetch PubChem SDF (3D if available, else 2D + embed)."""
    headers = {"User-Agent": USER_AGENT}
    url = PUBCHEM_SDF_URL.format(cid=cid)
    r = requests.get(url, params={"record_type": "3d"}, headers=headers, timeout=30)
    if r.status_code == 200 and r.text.strip():
        out_path.write_text(r.text, encoding="ascii", errors="ignore")
        return out_path

    r = requests.get(url, params={"record_type": "2d"}, headers=headers, timeout=30)
    r.raise_for_status()
    tmp_2d = out_path.with_suffix(".2d.sdf")
    tmp_2d.write_text(r.text, encoding="ascii", errors="ignore")
    suppl = Chem.SDMolSupplier(str(tmp_2d), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    tmp_2d.unlink(missing_ok=True)
    if mol is None:
        raise RuntimeError(f"PubChem CID {cid}: could not parse 2D SDF")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return out_path


def load_ligand(source: str, out_dir: str | Path) -> Path:
    """Return a 3D SDF (with explicit Hs) from a path, SMILES, or PubChem CID."""
    out_dir = _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(source).expanduser()
    if src_path.exists():
        src = _abs(src_path)
        suffix = src.suffix.lower()
        if suffix == ".sdf":
            dst = out_dir / src.name
            if src != dst:
                shutil.copy2(src, dst)
            return dst
        if suffix in (".mol2", ".pdb"):
            if suffix == ".mol2":
                mol = Chem.MolFromMol2File(str(src), removeHs=False)
            else:
                mol = Chem.MolFromPDBFile(str(src), removeHs=False)
            if mol is None:
                raise RuntimeError(f"rdkit could not parse {src}")
            if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
                mol = Chem.AddHs(mol, addCoords=True)
            dst = out_dir / (src.stem + ".sdf")
            writer = Chem.SDWriter(str(dst))
            writer.write(mol)
            writer.close()
            return dst
        raise ValueError(f"Unsupported ligand file extension: {suffix}")

    if source.isdigit():
        out_path = out_dir / f"cid_{source}.sdf"
        return _pubchem_fetch_sdf(source, out_path)

    # SMILES branch
    mol = Chem.MolFromSmiles(source)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {source!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise RuntimeError(f"ETKDG embedding failed for SMILES: {source!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:6]
    out_path = out_dir / f"ligand_{digest}.sdf"
    writer = Chem.SDWriter(str(out_path))
    writer.write(mol)
    writer.close()
    return out_path


def prepare_ligand_pdbqt(sdf_path: str | Path, out_pdbqt: str | Path) -> Path:
    """Convert a 3D SDF into a Vina-ready PDBQT via meeko."""
    sdf_path = _abs(sdf_path)
    out_pdbqt = _abs(out_pdbqt)
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    from meeko import MoleculePreparation  # lazy

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise RuntimeError(f"Could not read mol from {sdf_path}")
    if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)

    prep = MoleculePreparation()
    try:
        setups = prep.prepare(mol)
    except Exception:
        setups = None

    wrote = False
    if setups:
        try:
            from meeko import PDBQTWriterLegacy

            setup = setups[0] if isinstance(setups, (list, tuple)) else setups
            result = PDBQTWriterLegacy.write_string(setup)
            text = result[0] if isinstance(result, tuple) else result
            out_pdbqt.write_text(text)
            wrote = True
        except ImportError:
            pass
        except Exception:
            pass

    if not wrote:
        try:
            prep.write_pdbqt_file(str(out_pdbqt))
            wrote = True
        except Exception as exc:
            raise RuntimeError(
                "meeko could not write PDBQT for ligand; check meeko version"
            ) from exc

    return out_pdbqt


def prepare_receptor_pdbqt(
    pdb_path: str | Path,
    out_pdbqt: str | Path,
    keep_hetero: list[str] | None = None,
) -> Path:
    """Clean a receptor PDB, add polar Hs, write PDBQT via meeko (CLI fallback)."""
    pdb_path = _abs(pdb_path)
    out_pdbqt = _abs(out_pdbqt)
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    tmp_clean = out_pdbqt.with_suffix(".clean.pdb")
    clean_structure(pdb_path, tmp_clean)

    cmd.reinitialize()
    cmd.load(str(tmp_clean), "rec")
    if keep_hetero:
        keep_sel = " or ".join(f"resn {r}" for r in keep_hetero)
        cmd.remove(f"hetatm and not ({keep_sel})")
    else:
        cmd.remove("hetatm")
    cmd.h_add("polymer")
    tmp_prepped = out_pdbqt.with_suffix(".prepped.pdb")
    cmd.save(str(tmp_prepped), "rec")
    cmd.delete("all")

    wrote = False
    try:
        from meeko import PDBQTReceptor  # type: ignore

        rec = PDBQTReceptor(str(tmp_prepped))
        try:
            from meeko import PDBQTWriterLegacy

            text = PDBQTWriterLegacy.write_string(rec)
            text = text[0] if isinstance(text, tuple) else text
            out_pdbqt.write_text(text)
            wrote = True
        except Exception:
            try:
                rec.write_pdbqt_file(str(out_pdbqt))  # type: ignore[attr-defined]
                wrote = True
            except Exception:
                wrote = False
    except Exception:
        wrote = False

    if not wrote:
        exe = shutil.which("mk_prepare_receptor.py") or shutil.which("mk_prepare_receptor")
        if exe is None:
            raise RuntimeError(
                "Could not prepare receptor PDBQT: meeko Python API failed and "
                "mk_prepare_receptor.py is not on PATH. Install meeko CLI."
            )
        proc = subprocess.run(
            [exe, "-i", str(tmp_prepped), "-p", str(out_pdbqt)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_pdbqt.exists():
            raise RuntimeError(
                f"mk_prepare_receptor failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )

    tmp_clean.unlink(missing_ok=True)
    tmp_prepped.unlink(missing_ok=True)
    return out_pdbqt


def compute_box(
    pdb_path: str | Path,
    center_residues: list[tuple[str, int]],
    padding: float = 8.0,
) -> dict:
    """Return {center, size} for a Vina box around the given residues."""
    pdb_path = _abs(pdb_path)
    if not center_residues:
        raise ValueError("center_residues must be non-empty")

    cmd.reinitialize()
    cmd.load(str(pdb_path), "mol")
    selectors = [f"(chain {c} and resi {r})" for c, r in center_residues]
    sel = " or ".join(selectors)
    cmd.select("box_sel", sel)

    coords: list[tuple[float, float, float]] = []
    cmd.iterate_state(
        1,
        "box_sel",
        "coords.append((x, y, z))",
        space={"coords": coords},
    )
    cmd.delete("all")

    if not coords:
        raise RuntimeError(f"No atoms found for selection {sel!r}")

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    sx = (max(xs) - min(xs)) + 2 * padding
    sy = (max(ys) - min(ys)) + 2 * padding
    sz = (max(zs) - min(zs)) + 2 * padding
    return {
        "center": (round(cx, 3), round(cy, 3), round(cz, 3)),
        "size": (round(sx, 3), round(sy, 3), round(sz, 3)),
    }


_VINA_RESULT_RE = re.compile(
    r"REMARK\s+VINA\s+RESULT:\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)"
)


def _parse_vina_pdbqt(out_pdbqt: Path) -> pd.DataFrame:
    rows: list[dict] = []
    mode = 0
    text = out_pdbqt.read_text()
    for line in text.splitlines():
        if line.startswith("MODEL"):
            mode += 1
        m = _VINA_RESULT_RE.search(line)
        if m:
            rows.append(
                {
                    "mode": mode,
                    "affinity": float(m.group(1)),
                    "rmsd_lb": float(m.group(2)),
                    "rmsd_ub": float(m.group(3)),
                }
            )
    return pd.DataFrame(rows, columns=["mode", "affinity", "rmsd_lb", "rmsd_ub"])


def _parse_vina_stdout(stdout: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                rows.append(
                    {
                        "mode": int(parts[0]),
                        "affinity": float(parts[1]),
                        "rmsd_lb": float(parts[2]),
                        "rmsd_ub": float(parts[3]),
                    }
                )
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["mode", "affinity", "rmsd_lb", "rmsd_ub"])


def run_vina(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    box: dict,
    out_pdbqt: str | Path,
    exhaustiveness: int = 16,
    num_modes: int = 9,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Dock with AutoDock Vina (Python bindings preferred, CLI fallback)."""
    receptor_pdbqt = _abs(receptor_pdbqt)
    ligand_pdbqt = _abs(ligand_pdbqt)
    out_pdbqt = _abs(out_pdbqt)
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    center = tuple(float(c) for c in box["center"])
    size = tuple(float(s) for s in box["size"])

    try:
        from vina import Vina  # lazy

        v = Vina(sf_name="vina", seed=seed if seed is not None else 0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(center=list(center), box_size=list(size))
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        v.write_poses(str(out_pdbqt), n_poses=num_modes, overwrite=True)
        return _parse_vina_pdbqt(out_pdbqt)
    except ImportError:
        pass
    except Exception:
        # fall through to CLI
        pass

    exe = shutil.which("vina")
    if exe is None:
        raise RuntimeError(
            "Vina Python bindings not usable and `vina` binary not on PATH."
        )
    cmd_args = [
        exe,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size[0]),
        "--size_y", str(size[1]),
        "--size_z", str(size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--out", str(out_pdbqt),
    ]
    if seed is not None:
        cmd_args += ["--seed", str(seed)]
    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"vina failed: {proc.stderr.strip() or proc.stdout.strip()}")
    df = _parse_vina_pdbqt(out_pdbqt) if out_pdbqt.exists() else pd.DataFrame()
    if df.empty:
        df = _parse_vina_stdout(proc.stdout)
    return df


def split_poses(out_pdbqt: str | Path, out_dir: str | Path) -> list[Path]:
    """Split a multi-MODEL PDBQT into per-pose files."""
    out_pdbqt = _abs(out_pdbqt)
    out_dir = _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = out_pdbqt.read_text()
    lines = text.splitlines(keepends=True)
    poses: list[Path] = []
    buf: list[str] = []
    current_mode: int | None = None
    in_model = False
    for line in lines:
        if line.startswith("MODEL"):
            parts = line.split()
            try:
                current_mode = int(parts[1])
            except (IndexError, ValueError):
                current_mode = (poses[-1] if False else len(poses) + 1)
            buf = [line]
            in_model = True
            continue
        if in_model:
            buf.append(line)
            if line.startswith("ENDMDL"):
                n = current_mode if current_mode is not None else len(poses) + 1
                path = out_dir / f"{out_pdbqt.stem}_pose{n}.pdbqt"
                path.write_text("".join(buf))
                poses.append(path)
                buf = []
                in_model = False
                current_mode = None

    if not poses:
        # no MODEL records — treat the whole file as a single pose
        path = out_dir / f"{out_pdbqt.stem}_pose1.pdbqt"
        shutil.copy2(out_pdbqt, path)
        poses.append(path)
    return poses


def ligand_contacts(
    complex_pdb: str | Path,
    ligand_resn: str,
    cutoff: float = 4.0,
) -> pd.DataFrame:
    """Per-residue contacts from polymer to a ligand by residue name."""
    complex_pdb = _abs(complex_pdb)

    cmd.reinitialize()
    cmd.load(str(complex_pdb), "cx")

    lig_coords: list[tuple[float, float, float]] = []
    cmd.iterate_state(
        1,
        f"cx and resn {ligand_resn}",
        "coords.append((x, y, z))",
        space={"coords": lig_coords},
    )
    if not lig_coords:
        cmd.delete("all")
        return pd.DataFrame(columns=["chain", "resi", "resn", "min_dist", "n_contacts"])

    cmd.select("contact_res", f"byres (polymer within {cutoff} of (resn {ligand_resn}))")

    # cmd.iterate compiles its expression in single-statement mode, so an inline
    # `if` raises "multiple statements found"; collect every CA row and dedupe here.
    ca_rows: list[tuple[str, int, str]] = []
    cmd.iterate(
        "contact_res and name CA",
        "ca_rows.append((chain, int(resi), resn))",
        space={"ca_rows": ca_rows},
    )
    residues: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for chain, resi, resn in ca_rows:
        if (chain, resi) not in seen:
            seen.add((chain, resi))
            residues.append((chain, resi, resn))

    rows: list[dict] = []
    cutoff_sq = cutoff * cutoff
    for chain, resi, resn in residues:
        atoms: list[tuple[float, float, float]] = []
        cmd.iterate_state(
            1,
            f"contact_res and chain {chain} and resi {resi}",
            "coords.append((x, y, z))",
            space={"coords": atoms},
        )
        if not atoms:
            continue
        min_d_sq = float("inf")
        n_contacts = 0
        for ax, ay, az in atoms:
            for lx, ly, lz in lig_coords:
                d_sq = (ax - lx) ** 2 + (ay - ly) ** 2 + (az - lz) ** 2
                if d_sq < min_d_sq:
                    min_d_sq = d_sq
                if d_sq <= cutoff_sq:
                    n_contacts += 1
        rows.append(
            {
                "chain": chain,
                "resi": resi,
                "resn": resn,
                "min_dist": round(min_d_sq ** 0.5, 3),
                "n_contacts": n_contacts,
            }
        )

    cmd.delete("all")
    df = pd.DataFrame(rows, columns=["chain", "resi", "resn", "min_dist", "n_contacts"])
    return df.sort_values("min_dist").reset_index(drop=True)


def render_pose(
    receptor_pdb: str | Path,
    pose_pdbqt: str | Path,
    out_png: str | Path,
    contact_residues: list[tuple[str, int]] | None = None,
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Render receptor cartoon with a docked ligand pose as sticks."""
    receptor_pdb = _abs(receptor_pdb)
    pose_pdbqt = _abs(pose_pdbqt)
    out_png = _abs(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cmd.reinitialize()
    cmd.load(str(receptor_pdb), "rec")
    cmd.load(str(pose_pdbqt), "lig", format="pdbqt")
    cmd.hide("everything")
    cmd.show("cartoon", "rec")
    cmd.util.cbc("rec")
    cmd.show("sticks", "lig")
    cmd.util.cbao("lig")

    if contact_residues:
        selectors = [f"(rec and chain {c} and resi {r})" for c, r in contact_residues]
        sel = " or ".join(selectors)
        cmd.select("contacts", sel)
        cmd.show("sticks", "contacts")
        cmd.util.cbac("contacts")
        cmd.delete("contacts")

    _setup_view(width, height)
    cmd.orient("lig")
    cmd.zoom("lig", buffer=6.0)
    cmd.ray(width, height)
    cmd.png(str(out_png), dpi=150)
    cmd.delete("all")
    return out_png
