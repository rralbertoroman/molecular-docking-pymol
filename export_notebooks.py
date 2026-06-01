#!/usr/bin/env python
"""Export changed notebooks to PDF.

Detects the Jupyter notebooks that changed on the current branch (committed vs
``main`` plus uncommitted/untracked working-tree changes) and exports each to PDF
under ``exports_pdf/`` using nbconvert's ``webpdf`` engine (Playwright + chromium,
no LaTeX required). Notebooks are exported as-is — their already-saved outputs —
without re-execution.

Usage::

    uv run python export_notebooks.py                 # changed notebooks vs main
    uv run python export_notebooks.py --all           # every notebook in the repo
    uv run python export_notebooks.py --base develop  # change base ref
    uv run python export_notebooks.py a.ipynb b.ipynb # explicit notebooks
    uv run python export_notebooks.py --html          # fallback: export to HTML
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout (empty string on failure)."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
        return out.stdout if out.returncode == 0 else ""
    except FileNotFoundError:
        return ""


def repo_root() -> Path:
    top = _git(["rev-parse", "--show-toplevel"], Path.cwd()).strip()
    return Path(top) if top else Path.cwd()


def _is_notebook(rel: str, root: Path) -> bool:
    if not rel.endswith(".ipynb"):
        return False
    if ".ipynb_checkpoints/" in rel:
        return False
    return (root / rel).is_file()


def changed_notebooks(root: Path, base_ref: str) -> list[Path]:
    """Notebooks changed vs `base_ref` plus uncommitted/untracked ones."""
    found: set[str] = set()

    # committed changes on this branch vs the merge-base with base_ref
    base = _git(["merge-base", base_ref, "HEAD"], root).strip() or base_ref
    diff = _git(["diff", "--name-only", f"{base}..HEAD", "--", "*.ipynb"], root)
    found.update(line.strip() for line in diff.splitlines() if line.strip())

    # working tree: staged, unstaged, untracked (porcelain: "XY <path>")
    status = _git(["status", "--porcelain", "--", "*.ipynb"], root)
    for line in status.splitlines():
        if not line.strip():
            continue
        code, _, rest = line[:2], line[2], line[3:]
        if code.strip() == "D":  # deleted in both columns
            continue
        path = rest.split(" -> ")[-1].strip().strip('"')  # handle renames
        found.add(path)

    nbs = sorted(p for p in found if _is_notebook(p, root))
    return [root / p for p in nbs]


def all_notebooks(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts and ".venv" not in p.parts
    )


def export_one(nb: Path, root: Path, out_dir: Path, fmt: str) -> bool:
    """Convert one notebook; return True on success. Preserves relative dirs."""
    rel = nb.resolve().relative_to(root.resolve())
    target_dir = out_dir / rel.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "nbconvert", "--to", fmt]
    if fmt == "webpdf":
        cmd.append("--allow-chromium-download")
    cmd += ["--output-dir", str(target_dir), "--output", nb.stem, str(nb)]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"  ✗ {rel}\n{proc.stderr.strip()[-800:]}\n")
        return False
    ext = "html" if fmt == "html" else "pdf"
    print(f"  ✓ {rel}  ->  {(target_dir / (nb.stem + '.' + ext)).relative_to(root)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="*", help="explicit notebooks (overrides detection)")
    ap.add_argument("--all", action="store_true", help="export every notebook in the repo")
    ap.add_argument("--base", default="main", help="branch/ref to diff against (default: main)")
    ap.add_argument("--out-dir", default="exports_pdf", help="output directory (default: exports_pdf)")
    ap.add_argument("--html", action="store_true", help="fallback: export to HTML instead of webpdf")
    args = ap.parse_args()

    root = repo_root()
    out_dir = (root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    fmt = "html" if args.html else "webpdf"

    if args.notebooks:
        notebooks = [Path(n).resolve() for n in args.notebooks]
    elif args.all:
        notebooks = all_notebooks(root)
    else:
        notebooks = changed_notebooks(root, args.base)

    notebooks = [n for n in notebooks if n.is_file()]
    if not notebooks:
        print("No changed notebooks to export.")
        return 0

    print(f"Exporting {len(notebooks)} notebook(s) to {out_dir.relative_to(root) if out_dir.is_relative_to(root) else out_dir}/ as {fmt}:")
    failures = 0
    for nb in notebooks:
        if not export_one(nb, root, out_dir, fmt):
            failures += 1

    print(f"\nDone: {len(notebooks) - failures}/{len(notebooks)} exported. Output in {out_dir}/")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
