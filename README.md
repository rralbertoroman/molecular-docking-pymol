# molecular-docking-pymol

Automatización del flujo PyMOL del trabajo final de Bioinformática (3er año
Bioquímica) — cubre los puntos 3 (acoplamiento molecular) y 4 (residuos de
interfaz) de las orientaciones en [`guidelines/Trabajo_final.pdf`](guidelines/Trabajo_final.pdf).

## Estructura

- `pymol_utils.py` — funciones reutilizables de PyMOL (fetch, clean, render,
  interface, align).
- `pymol_workflow.ipynb` — notebook para **docking proteína–proteína**
  (compañero de la práctica CP3 / ClusPro). Secciones: Setup / Prepare / Pair /
  Analyze.
- `ligand_utils.py` — utilidades para **docking proteína–ligando pequeño** con
  AutoDock Vina (load_ligand, prepare_*_pdbqt, compute_box, run_vina,
  split_poses, ligand_contacts, render_pose).
- `ligand_docking.ipynb` — notebook para docking de **cefoxitina contra PBP-6
  (UniProt P08506, PDB 3IT9)** como caso de uso por defecto. Editable a
  cualquier receptor + ligando.

## Instalación

```bash
uv sync
```

Esto instala `pymol-open-source`, `rdkit`, `meeko`, `pandas`, `requests`,
`jupyter`.

### AutoDock Vina

El binario de Vina **no se instala automáticamente**. Para correr la sección 4
del notebook `ligand_docking.ipynb` necesitas Vina en el PATH. Opciones:

- **Debian/Ubuntu**: `sudo apt install autodock-vina` (v1.1.2, suficiente).
- **Release oficial**: descarga el binario estático desde
  https://github.com/ccsb-scripps/AutoDock-Vina/releases (v1.2.5) y ponlo en
  `~/.local/bin/vina`.
- **conda**: `conda install -c conda-forge autodock-vina`.

`ligand_utils.run_vina` detecta el binario en el PATH y, si fallan las
bindings Python, hace fallback a subprocess.

## Uso rápido

```bash
uv run jupyter lab pymol_workflow.ipynb       # protein–protein
uv run jupyter lab ligand_docking.ipynb       # protein–small ligand
```

Cada notebook tiene una celda **Setup** al principio: edita las variables
(PDB IDs, paths, residuos del bolsillo) y corre el resto en orden.

## Notas

- Para `ligand_docking.ipynb` con un receptor distinto a 3IT9, **actualiza
  `POCKET_RESIDUES`** con la numeración del PDB elegido (no la de UniProt — los
  PDBs frecuentemente recortan el péptido señal y/o renumeran).
- El docking de cefoxitina con Vina es no covalente; la pose representa la
  aproximación pre-acilación, no el aducto acil-enzima final.
