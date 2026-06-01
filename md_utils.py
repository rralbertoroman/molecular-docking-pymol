"""Apo-protein molecular-dynamics helpers (OpenMM) + trajectory analysis.

Companion to ligand_utils / pymol_utils. Runs a short implicit-solvent MD of the
*receptor only* (the docked ligand is not parameterized — openff-toolkit is
unavailable on PyPI and AmberTools is not pip-installable), then derives RMSD,
RMSF and Rg with MDAnalysis. The run is deliberately short and illustrative.
"""

from __future__ import annotations

from pathlib import Path


def _abs(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _select_platform(name: str = "auto"):
    """Return (Platform, properties). 'auto' tries CUDA, then OpenCL, then CPU."""
    import openmm as mm

    if name and name != "auto":
        plat = mm.Platform.getPlatformByName(name)
        props = {"Precision": "mixed"} if name in ("CUDA", "OpenCL") else {}
        return plat, props
    for cand in ("CUDA", "OpenCL"):
        try:
            return mm.Platform.getPlatformByName(cand), {"Precision": "mixed"}
        except Exception:
            continue
    return mm.Platform.getPlatformByName("CPU"), {}


def prepare_md_system(pdb_in: str | Path, out_pdb: str | Path, ph: float = 7.0) -> Path:
    """Complete + protonate a receptor PDB with PDBFixer for MD.

    Only fills missing heavy atoms in *resolved* residues and adds hydrogens;
    unresolved terminal/loop residues are intentionally NOT modeled
    (`missingResidues = {}`).
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    pdb_in = _abs(pdb_in)
    out_pdb = _abs(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    # original residue numbering (PDBFixer renumbers to 1..N; we restore it so
    # the MD keeps the 3IT9 numbering used everywhere else in the notebook)
    orig_ids = [(r.chain.id, r.id) for r in PDBFile(str(pdb_in)).topology.residues()]

    fixer = PDBFixer(filename=str(pdb_in))
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # do not build unresolved termini/loops
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    # restore original residue ids (1:1 mapping — no residues added/removed here)
    residues = list(fixer.topology.residues())
    if len(residues) == len(orig_ids):
        for res, (_, rid) in zip(residues, orig_ids):
            res.id = rid

    with open(out_pdb, "w") as fh:
        # keepIds=True preserves the 3IT9 residue numbering (default renumbers 1..N)
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    return out_pdb


def run_md(
    fixed_pdb: str | Path,
    out_dir: str | Path,
    prod_ps: float = 500.0,
    equil_ps: float = 100.0,
    temperature_K: float = 300.0,
    timestep_fs: float = 2.0,
    report_ps: float = 2.0,
    seed: int = 42,
    minimize_iters: int = 500,
    platform: str = "auto",
    force: bool = False,
) -> dict:
    """Minimize -> equilibrate -> production MD (Amber ff14SB + implicit OBC2).

    Writes a DCD trajectory, a topology PDB, and a StateData CSV (potential/kinetic
    energy, temperature). Cache-aware: skips if the trajectory already exists.
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    fixed_pdb = _abs(fixed_pdb)
    out_dir = _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj = out_dir / "traj.dcd"
    log = out_dir / "md_log.csv"
    topo = out_dir / "md_topology.pdb"

    meta = {
        "trajectory": traj,
        "log": log,
        "topology": topo,
        "report_ps": report_ps,
        "temperature_K": temperature_K,
    }
    if traj.exists() and log.exists() and topo.exists() and not force:
        return meta

    pdb = app.PDBFile(str(fixed_pdb))
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=app.HBonds,
    )
    dt = timestep_fs * unit.femtoseconds
    integrator = mm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin, 1.0 / unit.picosecond, dt
    )
    integrator.setRandomNumberSeed(int(seed))
    plat, plat_props = _select_platform(platform)
    meta["platform"] = plat.getName()
    sim = app.Simulation(pdb.topology, system, integrator, plat, plat_props)
    sim.context.setPositions(pdb.positions)

    # cap iterations: GB-implicit minimization is O(N^2) and slow on CPU
    sim.minimizeEnergy(maxIterations=minimize_iters)

    # topology snapshot (atom order matches the DCD frames)
    with open(topo, "w") as fh:
        app.PDBFile.writeFile(
            sim.topology,
            sim.context.getState(getPositions=True).getPositions(),
            fh,
            keepIds=True,
        )

    steps_per_ps = int(round(1000.0 / timestep_fs))
    report_steps = max(1, int(round(report_ps * steps_per_ps)))
    equil_steps = int(round(equil_ps * steps_per_ps))
    prod_steps = int(round(prod_ps * steps_per_ps))

    sim.context.setVelocitiesToTemperature(temperature_K * unit.kelvin, int(seed))
    if equil_steps > 0:
        sim.step(equil_steps)

    sim.reporters.append(app.DCDReporter(str(traj), report_steps))
    sim.reporters.append(
        app.StateDataReporter(
            str(log),
            report_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
        )
    )
    sim.step(prod_steps)
    return meta


def analyze_md(
    topology_pdb: str | Path,
    traj_dcd: str | Path,
    out_dir: str | Path,
    frame_dt_ps: float = 2.0,
    active_site: tuple[int, ...] = (40, 43, 106, 209),
) -> dict:
    """Compute backbone RMSD(t), per-residue Cα RMSF, and Rg(t) with MDAnalysis.

    Writes md_rmsd.csv, md_rmsf.csv, md_rg.csv. Returns the three DataFrames.
    `frame_dt_ps` is the spacing between saved frames (matches run_md report_ps).
    """
    import numpy as np
    import pandas as pd
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms, align

    out_dir = _abs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(str(_abs(topology_pdb)), str(_abs(traj_dcd)))
    n_frames = len(u.trajectory)

    # RMSD of protein backbone vs first frame (RMSD superimposes internally)
    R = rms.RMSD(u, u, select="backbone", ref_frame=0).run()
    rmsd = R.results.rmsd  # cols: frame, time, rmsd(A)
    rmsd_df = pd.DataFrame(
        {
            "frame": rmsd[:, 0].astype(int),
            "time_ps": rmsd[:, 0] * frame_dt_ps,
            "rmsd_A": rmsd[:, 2],
        }
    )
    rmsd_df.to_csv(out_dir / "md_rmsd.csv", index=False)

    # Rg of the protein per frame (rotation-invariant; safe before/after align)
    prot = u.select_atoms("protein")
    rg_rows = []
    for i, _ in enumerate(u.trajectory):
        rg_rows.append((i * frame_dt_ps, float(prot.radius_of_gyration())))
    rg_df = pd.DataFrame(rg_rows, columns=["time_ps", "rg_A"])
    rg_df.to_csv(out_dir / "md_rg.csv", index=False)

    # per-residue Cα RMSF (requires aligning the trajectory first)
    align.AlignTraj(u, u, select="backbone", ref_frame=0, in_memory=True).run()
    ca = u.select_atoms("protein and name CA")
    F = rms.RMSF(ca).run()
    rmsf_df = pd.DataFrame(
        {
            "resid": ca.resids,
            "resname": ca.resnames,
            "rmsf_A": F.results.rmsf,
            "is_active_site": np.isin(ca.resids, np.array(active_site)),
        }
    )
    rmsf_df.to_csv(out_dir / "md_rmsf.csv", index=False)

    return {"rmsd": rmsd_df, "rmsf": rmsf_df, "rg": rg_df, "n_frames": n_frames}
