# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "rdkit",
#   "ase",
# ]
# ///
"""
Multi-conformer CSP driver: mini-CSP conformer ranking, then full CSP.

Usage:
    python confgnrs.py -c inp.conf --stage mini   # conformers -> micro-CSP -> rank -> cf1..cfN/
    python confgnrs.py -c inp.conf --stage full   # run gnrs in every cfN/ -> rank -> merge
    python confgnrs.py -c inp.conf --stage all    # both

Stage "mini":
  1. Systematic dihedral grid (n_angles per heavy-atom rotatable bond),
     xTB GFN2 single-point, keep 2*max_conformers within energy_window
  2. conformers/micro-cfX/: gnrs with only micro_csp_spgs and
     num_structures_per_spg from [conformer]
  3. xTB GFN1 periodic single-point on each dedup'd crystal, rank conformers
     by lowest E/molecule, write top max_conformers to cf1/ .. cfN/
Stage "full":
  Run gnrs in each cfN/ with the user's [generation] settings (skips cfN/
  that already finished), rank by best press_energy, and merge every cfN's
  dedup structures into merged_structures.json.

Config file (inp.conf); all sections except [conformer]/[workflow] are passed
through to each gnrs run, workflow is fixed to generation -> symm_rigid_press -> dedup:
    [conformer]
    mol                     = mol.xyz
    n_angles                = 5          # dihedral angles per rotatable bond
    energy_window           = 20.0       # kcal/mol
    max_conformers          = 20         # conformers sent to full CSP
    charge                  = 0
    num_structures_per_spg  = 20         # mini-CSP only
    micro_csp_spgs          = 14,2,19    # mini-CSP only
    mpi_np                  = 4          # MPI ranks per gnrs run / xTB workers
    gnrs_cmd                = gnrs       # optional; default: .venv or gnrs_env gnrs
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from configparser import ConfigParser

warnings.filterwarnings("ignore", category=FutureWarning, module="ase")

KCAL_PER_HARTREE = 627.509474
WIDTH = 80
HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = "['generation', 'symm_rigid_press', 'dedup']"
# One thread per process: many xTB / MPI ranks run concurrently
ENV_1THREAD = {v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}


def banner(title):
    print("=" * WIDTH + f"\n  {title}\n" + "=" * WIDTH, flush=True)


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ── Config ────────────────────────────────────────────────────────────────

def load_config(config_path):
    """Read inp.conf; return (settings_dict, raw_ConfigParser)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cp = ConfigParser()
    cp.read(config_path)
    c = cp["conformer"] if cp.has_section("conformer") else {}
    cfg = {
        "mol":                    c.get("mol", "mol.xyz"),
        "n_angles":               int(c.get("n_angles", 5)),
        "energy_window":          float(c.get("energy_window", 20.0)),
        "max_conformers":         int(c.get("max_conformers", 20)),
        "charge":                 int(c.get("charge", 0)),
        "num_structures_per_spg": int(c.get("num_structures_per_spg", 20)),
        "mpi_np":                 int(c.get("mpi_np", 4)),
        # Most common organic space groups by CSD frequency:
        # P2₁/c (#14, ~36%), P-1 (#2, ~16%), P2₁2₁2₁ (#19, ~11%)
        "micro_csp_spgs":         c.get("micro_csp_spgs", "14,2,19"),
        "gnrs_cmd":               c.get("gnrs_cmd", _default_gnrs()),
        "z":                      int(cp.get("master", "z", fallback="1")),
    }
    return cfg, cp


def _default_gnrs():
    for p in (".venv/bin/gnrs", "gnrs_env/bin/gnrs"):
        if os.path.exists(os.path.join(HERE, p)):
            return os.path.join(HERE, p)
    return "gnrs"


def write_conf(src_cp, name, out_path, generation=None):
    """Write a gnrs inp.conf: passthrough sections, fixed workflow, [generation] overrides."""
    out = ConfigParser()
    for section in src_cp.sections():
        if section not in ("conformer", "workflow"):
            out[section] = dict(src_cp.items(section))
    out.setdefault("master", {})
    out["master"].update(name=name, molecule_path='["mol.xyz"]')
    out["workflow"] = {"tasks": TASKS}
    if generation:
        out.setdefault("generation", {})
        out["generation"].update(generation)
    with open(out_path, "w") as fh:
        out.write(fh)


# ── gnrs runs ─────────────────────────────────────────────────────────────

def dedup_structs(folder):
    path = os.path.join(folder, "structures", "dedup", "structures.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run_gnrs_dirs(folders, cfg):
    """Run `mpirun -n mpi_np gnrs -c inp.conf` in each folder, one at a time."""
    # Stdout must go to a file: OpenMPI stdio forwarding deadlocks on pipes.
    # Drop the parent's venv vars so the gnrs launcher uses its own env.
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONNOUSERSITE")}
    env.update(ENV_1THREAD)
    cmd = ["mpirun", "-n", str(cfg["mpi_np"]), cfg["gnrs_cmd"], "-c", "inp.conf"]
    print(f"  {' '.join(cmd)}  (per folder)")
    failed = []
    for i, folder in enumerate(folders, 1):
        name = os.path.basename(folder)
        print(f"  [{i}/{len(folders)}] {name:<12}", end="", flush=True)
        if dedup_structs(folder) is not None:
            print("  already done, skipped")
            continue
        t = time.time()
        with open(os.path.join(folder, "run.log"), "w") as log:
            rc = subprocess.run(cmd, cwd=folder, env=env, stdout=log, stderr=log).returncode
        print(f"  {'OK' if rc == 0 else f'FAILED (rc={rc})'}  ({fmt_time(time.time() - t)})")
        if rc != 0:
            failed.append(name)
    if failed:
        print(f"  WARNING: {len(failed)} run(s) failed: {', '.join(failed)}")


# ── xTB ───────────────────────────────────────────────────────────────────

def parse_xtb_energy(stdout):
    energy = None
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                energy = float(line.split()[-3])
            except (ValueError, IndexError):
                continue
    if energy is None:
        raise RuntimeError("Could not parse xTB energy")
    return energy


def _xtb_sp(job):
    """Worker: xTB single-point on `fname` in `cwd`; returns energy (Eh) or None."""
    cwd, fname, flags, charge = job
    r = subprocess.run(["xtb", fname, *flags, "--scc", "--chrg", str(charge), "--norestart"],
                       cwd=cwd, capture_output=True, text=True, env={**os.environ, **ENV_1THREAD})
    try:
        return parse_xtb_energy(r.stdout)
    except RuntimeError:
        return None


def _decode_ndarray(obj):
    """Recursively decode ASE JSON ndarray objects."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            import numpy as np
            shape, dtype, data = obj["__ndarray__"]
            return np.array(data, dtype=dtype).reshape(shape)
        return {k: _decode_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_ndarray(v) for v in obj]
    return obj


# ── Stage mini ────────────────────────────────────────────────────────────

def sample_and_filter(mol_path, n_angles, energy_window, max_conformers, charge, work_dir,
                      n_workers=1):
    """Dihedral-grid conformers + xTB GFN2 SP; return (paths, energies) sorted by energy,
    uniformly sampled down to max_conformers within energy_window (kcal/mol)."""
    import numpy as np
    from ase.io import read as ase_read, write as ase_write
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds, rdMolTransforms
    from rdkit.Chem.rdmolfiles import MolToXYZBlock

    xyz_path = os.path.join(work_dir, "input.xyz")
    ase_write(xyz_path, ase_read(mol_path), format="xyz")
    mol = Chem.MolFromXYZFile(xyz_path)
    if mol is None:
        raise RuntimeError(f"RDKit could not read molecule from {xyz_path}")
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    # Heavy-atom rotatable bonds (skip pure H rotors like OH, NH2)
    quads = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1.0:
            continue
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ji = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()
              if a.GetIdx() != k and a.GetAtomicNum() != 1]
        ki = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors()
              if a.GetIdx() != j and a.GetAtomicNum() != 1]
        if ji and ki:
            quads.append((ji[0], j, k, ki[0]))

    angles = [i * 360.0 / n_angles for i in range(n_angles)]
    bonds = ", ".join(f"{mol.GetAtomWithIdx(b).GetSymbol()}{b}-{mol.GetAtomWithIdx(c).GetSymbol()}{c}"
                      for _, b, c, _ in quads)
    print(f"  Rotatable bonds ({len(quads)}): {bonds}")
    print(f"  Grid: {n_angles}^{len(quads)} = {n_angles ** len(quads)} conformers, "
          f"xTB GFN2 SP on {n_workers} workers", flush=True)

    sp_dir = os.path.join(work_dir, "sp")
    jobs = []
    for idx, combo in enumerate(itertools.product(angles, repeat=len(quads))):
        m = Chem.RWMol(mol)  # fresh copy of the base geometry per grid point
        for q, angle in zip(quads, combo):
            rdMolTransforms.SetDihedralDeg(m.GetConformer(0), *q, angle)
        cwd = os.path.join(sp_dir, f"conf_{idx:05d}")
        os.makedirs(cwd, exist_ok=True)
        with open(os.path.join(cwd, "mol.xyz"), "w") as fh:
            fh.write(MolToXYZBlock(m, confId=0))
        jobs.append((cwd, "mol.xyz", ["--gfn", "2"], charge))

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        energies = list(ex.map(_xtb_sp, jobs, chunksize=4))
    paired = sorted((e, os.path.join(j[0], "mol.xyz")) for e, j in zip(energies, jobs) if e is not None)
    if not paired:
        raise RuntimeError("All xTB single-points failed.")
    print(f"  xTB: {len(paired)} ok, {len(jobs) - len(paired)} failed")

    e_min = paired[0][0]
    window = [p for p in paired if (p[0] - e_min) * KCAL_PER_HARTREE <= energy_window]
    if len(window) > max_conformers:
        idx = np.round(np.linspace(0, len(window) - 1, max_conformers)).astype(int)
        window = [window[i] for i in idx]
    print(f"  {len(window)} conformers selected within {energy_window:.1f} kcal/mol")
    return [p for _, p in window], [e for e, _ in window]


def _xtb_crystal_job(calc_dir, struct_data, charge):
    from ase import Atoms
    from ase.io import write as ase_write
    s = _decode_ndarray(struct_data)
    os.makedirs(calc_dir, exist_ok=True)
    atoms = Atoms(numbers=s["numbers"], positions=s["positions"], cell=s["cell"], pbc=s["pbc"])
    ase_write(os.path.join(calc_dir, "geometry.in"), atoms, format="aims")
    return calc_dir, "geometry.in", ["--gfn", "1", "--periodic"], charge


def stage_mini(cfg, cp, run_dir):
    t0 = time.time()
    mol_path = os.path.join(run_dir, cfg["mol"])
    if not os.path.exists(mol_path):
        raise FileNotFoundError(f"Molecule file not found: {mol_path}")
    work_dir = os.path.join(run_dir, "_conformer_work")
    conf_dir = os.path.join(run_dir, "conformers")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    banner("Mini 1: Conformer sampling")
    paths, energies = sample_and_filter(
        mol_path, cfg["n_angles"], cfg["energy_window"], 2 * cfg["max_conformers"],
        cfg["charge"], work_dir, n_workers=cfg["mpi_np"],
    )
    t1 = time.time()

    banner(f"Mini 2: micro-CSP (spgs {cfg['micro_csp_spgs']}, "
           f"{cfg['num_structures_per_spg']}/spg)")
    conf_paths, micro_dirs = [], []
    gen = {"num_structures_per_spg": str(cfg["num_structures_per_spg"]),
           "spg_distribution_type": f"[{cfg['micro_csp_spgs']}]"}
    for i, src in enumerate(paths):
        conf = os.path.join(conf_dir, f"conf_{i:04d}.xyz")
        folder = os.path.join(conf_dir, f"micro-cf{i + 1}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(src, conf)
        shutil.copy(src, os.path.join(folder, "mol.xyz"))
        write_conf(cp, f"micro-cf{i + 1}", os.path.join(folder, "inp.conf"), gen)
        conf_paths.append(conf)
        micro_dirs.append(folder)
    run_gnrs_dirs(micro_dirs, cfg)
    t2 = time.time()

    banner(f"Mini 3: Crystal energy (xTB GFN1 periodic, {cfg['mpi_np']} workers)")
    owners, jobs = [], []
    for i, folder in enumerate(micro_dirs):
        for j, sdata in enumerate((dedup_structs(folder) or {}).values()):
            calc = os.path.join(folder, "_xtb_crystal", f"struct_{j:04d}")
            jobs.append(_xtb_crystal_job(calc, sdata, cfg["charge"]))
            owners.append(i)
    print(f"  {len(jobs)} structures")
    with ProcessPoolExecutor(max_workers=cfg["mpi_np"]) as ex:
        energies_xtal = list(ex.map(_xtb_sp, jobs))
    best = {}
    for i, e in zip(owners, energies_xtal):
        if e is not None and e / cfg["z"] < best.get(i, float("inf")):
            best[i] = e / cfg["z"]
    if not best:
        print("  WARNING: all xTB calculations failed, cannot rank conformers.")
        return

    ranked = sorted(best.items(), key=lambda x: x[1])
    e_min = ranked[0][1]
    top = ranked[: cfg["max_conformers"]]
    print(f"\n  {'Rank':<6}{'Conformer':<14}{'E/mol (Eh)':>14}{'ΔE (kcal/mol)':>16}  full CSP dir")
    for rank, (i, e) in enumerate(ranked, 1):
        cf = f"cf{rank}" if rank <= len(top) else "-"
        print(f"  {rank:<6}{f'micro-cf{i + 1}':<14}{e:>14.6f}{(e - e_min) * KCAL_PER_HARTREE:>16.3f}  {cf}")
    for rank, (i, _) in enumerate(top, 1):
        cf_dir = os.path.join(run_dir, f"cf{rank}")
        os.makedirs(cf_dir, exist_ok=True)
        shutil.copy(conf_paths[i], os.path.join(cf_dir, "mol.xyz"))
        write_conf(cp, f"cf{rank}", os.path.join(cf_dir, "inp.conf"))
    print(f"\n  Created cf1/ .. cf{len(top)}/")
    print(f"  Time: sampling {fmt_time(t1 - t0)}, micro-CSP {fmt_time(t2 - t1)}, "
          f"xTB {fmt_time(time.time() - t2)}")


# ── Stage full ────────────────────────────────────────────────────────────

def stage_full(cfg, cp, run_dir):
    cf_dirs = sorted((d for d in os.listdir(run_dir) if re.fullmatch(r"cf\d+", d)),
                     key=lambda d: int(d[2:]))
    if not cf_dirs:
        raise FileNotFoundError(f"No cfN/ folders in {run_dir}; run --stage mini first")
    banner(f"Full CSP: {len(cf_dirs)} conformers")
    t0 = time.time()
    run_gnrs_dirs([os.path.join(run_dir, d) for d in cf_dirs], cfg)

    merged, rows = {}, []
    for cf in cf_dirs:
        structs = dedup_structs(os.path.join(run_dir, cf))
        if structs is None:
            continue
        e = [s.get("info", {}).get("press_energy") for s in structs.values()]
        e = [x for x in e if x is not None]
        rows.append((min(e) if e else float("inf"), cf, len(structs)))
        for key, entry in structs.items():
            entry.setdefault("info", {}).update(
                conformer_id=cf, conformer_mol_path=os.path.join(run_dir, cf, "mol.xyz"))
            merged[f"{cf}__{key}"] = entry

    print(f"\n  {'Rank':<6}{'Conformer':<12}{'best press_energy':>18}{'n_unique':>10}")
    for rank, (e, cf, n) in enumerate(sorted(rows), 1):
        print(f"  {rank:<6}{cf:<12}{e:>18.4f}{n:>10}")
    out_path = os.path.join(run_dir, "merged_structures.json")
    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"\n  Merged {len(merged)} structures from {len(rows)} conformers -> {out_path}")
    print(f"  Time: {fmt_time(time.time() - t0)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("-c", "--config", required=True, help="Path to inp.conf")
    p.add_argument("--stage", choices=("mini", "full", "all"), default="mini")
    args = p.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg, cp = load_config(cfg_path)
    run_dir = os.path.dirname(cfg_path)
    if args.stage in ("mini", "all"):
        stage_mini(cfg, cp, run_dir)
    if args.stage in ("full", "all"):
        stage_full(cfg, cp, run_dir)


if __name__ == "__main__":
    main()
