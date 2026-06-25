# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "rdkit",
#   "ase",
# ]
# ///
"""
Conformer generation + gnrs mini-CSP folder setup.

Steps:
  1. ETKDG sampling + xTB GFN2 optimization → filter 0-energy_window kcal/mol
  2. Write surviving conformers to conformers/conf_NNNN.xyz
  3. Create conformers/micro-cf1/ ... conformers/micro-cfN/ (temporary mini-CSP dirs):
       mol.xyz   — the conformer geometry
       inp.conf  — gnrs mini-CSP config (passthrough, num_structures_per_spg=20)
  4. Run `mpirun -n mpi_np gnrs -c inp.conf` in each micro-cfX/, log → micro-cfX/run.log
  (Later) rank by crystal packing energy → create cf1/ cf2/ ... for full CSP

Usage:
    uv run confgnrs.py -c inp.conf

Config file format (inp.conf):
    [conformer]
    mol                     = mol.xyz
    n_candidates            = 200
    energy_window           = 10.0
    max_conformers          = 20
    pruning_rms             = 0.5
    charge                  = 0
    seed                    = 42
    num_structures_per_spg  = 20   # used for mini-CSP (overrides [generation])
    mpi_np                  = 4    # MPI processes per mini-CSP run

    [master]                       # passed through to cfX/inp.conf verbatim
    Z                       = 1
    log_level               = info

    [workflow]
    tasks                   = ['generation', 'symm_rigid_press', 'dedup']

    [generation]
    num_structures_per_spg  = 1000  # for normal full CSP (NOT for mini-CSP)
    ...

    [symm_rigid_press]
    ...

    [dedup]
    ...
"""

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser

KCAL_PER_HARTREE = 627.509474
WIDTH = 80

# 3ob-3-1 DFTB paths
_3OB_SK_FILES = "/home/xchen/Test/Genarris/test/asprin/potential/3ob-3-1"


def parse_args():
    p = argparse.ArgumentParser(description="Conformer generation + gnrs folder setup")
    p.add_argument("-c", "--config", required=True, help="Path to inp.conf")
    return p.parse_args()


def load_config(config_path: str):
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
        "seed":                   int(c.get("seed", 42)),
        "num_structures_per_spg": int(c.get("num_structures_per_spg", 20)),
        "mpi_np":                 int(c.get("mpi_np", 4)),
    }
    return cfg, cp


_MICRO_TASKS = "['generation', 'symm_rigid_press', 'dedup']"
_SKIP_SECTIONS = {"conformer", "workflow"}


def write_cf_conf(src_cp: ConfigParser, folder_name: str, num_spg: int, out_path: str):
    """Write micro-cfX/inp.conf.

    - [master]: passthrough + override name/molecule_path
    - [workflow]: hardcoded to generation → symm_rigid_press → dedup
    - [generation]: passthrough + override num_structures_per_spg
    - [symm_rigid_press], [dedup], others: passthrough verbatim
    - [conformer]: skipped
    """
    out = ConfigParser()

    for section in src_cp.sections():
        if section in _SKIP_SECTIONS:
            continue
        out.add_section(section)
        for key, val in src_cp.items(section):
            out.set(section, key, val)

    # [master] overrides
    if not out.has_section("master"):
        out.add_section("master")
    out.set("master", "name", folder_name)
    out.set("master", "molecule_path", '["mol.xyz"]')

    # fixed micro-CSP workflow
    out.add_section("workflow")
    out.set("workflow", "tasks", _MICRO_TASKS)

    # [generation] override
    if not out.has_section("generation"):
        out.add_section("generation")
    out.set("generation", "num_structures_per_spg", str(num_spg))

    with open(out_path, "w") as fh:
        out.write(fh)


_DFTB_SETTINGS_SRC = "/home/xchen/Test/Genarris/test/asprin/dftb_settings.json"


_CSP_TASKS = "['generation', 'symm_rigid_press', 'dedup', 'bfgs_dftb', 'dedup']"


def write_csp_conf(src_cp: ConfigParser, folder_name: str, out_path: str):
    """Write cf1/.../cfN/inp.conf for formal full CSP.
    Passthrough all sections except [conformer] and [workflow].
    Workflow is hardcoded to: generation → symm_rigid_press → bfgs_dftb → dedup.
    Adds [bfgs]/[dftb] if absent.
    """
    out = ConfigParser()
    for section in src_cp.sections():
        if section in ("conformer", "workflow"):
            continue
        out.add_section(section)
        for key, val in src_cp.items(section):
            out.set(section, key, val)
    if not out.has_section("master"):
        out.add_section("master")
    out.set("master", "name", folder_name)
    out.set("master", "molecule_path", '["mol.xyz"]')

    # Fixed CSP workflow: geometric pre-press then DFTB+ relaxation
    out.add_section("workflow")
    out.set("workflow", "tasks", _CSP_TASKS)

    # Add DFTB+ calculator sections if not already in user conf
    if not out.has_section("bfgs"):
        out.add_section("bfgs")
        out.set("bfgs", "energy_method", "dftb")
        out.set("bfgs", "fmax", "0.01")
        out.set("bfgs", "steps", "500")
        out.set("bfgs", "cell_opt", "True")
        out.set("bfgs", "fix_sym", "True")
    if not out.has_section("dftb"):
        out.add_section("dftb")
        out.set("dftb", "command", "dftb+ > dftb.out")
        out.set("dftb", "sk_files", _3OB_SK_FILES)
        out.set("dftb", "energy_settings_path", "./dftb_settings.json")

    with open(out_path, "w") as fh:
        out.write(fh)


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


def _xtb_crystal_sp(args):
    """Worker: xTB GFN1 periodic single-point on one crystal structure."""
    calc_dir, struct_data, charge = args
    from ase import Atoms
    from ase.io import write as ase_write

    os.makedirs(calc_dir, exist_ok=True)
    s = _decode_ndarray(struct_data)
    atoms = Atoms(
        numbers=s["numbers"],
        positions=s["positions"],
        cell=s["cell"],
        pbc=s["pbc"],
    )
    ase_write(os.path.join(calc_dir, "geometry.in"), atoms, format="aims")

    result = subprocess.run(
        ["xtb", "geometry.in", "--gfn", "1", "--scc",
         "--chrg", str(charge), "--periodic", "--norestart"],
        cwd=calc_dir, capture_output=True, text=True,
    )
    try:
        return _parse_xtb_energy(result.stdout)
    except RuntimeError:
        return None


def rank_and_create_cf_dirs(micro_dirs, conf_paths, run_dir, raw_cp, cfg):
    """Step 4+5: xTB crystal single-points → rank conformers → create cf1/...cfN/."""
    Z = int(raw_cp.get("master", "z", fallback="1"))
    charge = cfg["charge"]
    n_workers = cfg["mpi_np"]

    print("=" * WIDTH)
    print("  Step 4: Crystal Packing Energy (xTB GFN2 periodic)")
    print(f"  {n_workers} parallel workers")
    print("=" * WIDTH)

    # Collect all (micro_cf_idx, struct_idx, hash_id, struct_data) jobs
    jobs = []
    for i, folder in enumerate(micro_dirs):
        struct_json = os.path.join(folder, "structures", "dedup", "structures.json")
        if not os.path.exists(struct_json):
            continue
        with open(struct_json) as f:
            structs = json.load(f)
        xtb_dir = os.path.join(folder, "_xtb_crystal")
        for j, (hash_id, sdata) in enumerate(structs.items()):
            calc_dir = os.path.join(xtb_dir, f"struct_{j:04d}")
            jobs.append((i, j, calc_dir, sdata))

    total = len(jobs)
    print(f"  Total structures to evaluate: {total}")
    print()

    # Run in parallel
    best_per_cf = {}  # micro_cf_idx → best E/mol (Eh)
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(_xtb_crystal_sp, (calc_dir, sdata, charge)): (i, j)
            for i, j, calc_dir, sdata in jobs
        }
        for fut in as_completed(futs):
            i, j = futs[fut]
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}]", flush=True)
            e = fut.result()
            if e is None:
                continue
            e_per_mol = e / Z
            if i not in best_per_cf or e_per_mol < best_per_cf[i]:
                best_per_cf[i] = e_per_mol

    print()
    if not best_per_cf:
        print("  WARNING: All xTB calculations failed. Cannot rank conformers.")
        return

    # Rank conformers by best crystal packing energy
    ranked = sorted(best_per_cf.items(), key=lambda x: x[1])
    e_min = ranked[0][1]

    print(f"  {'Rank':<6} {'Conformer':<14} {'E/mol (Eh)':>14}  {'ΔE (kcal/mol)':>14}")
    print("  " + "-" * 52)
    for rank, (idx, e) in enumerate(ranked):
        de = (e - e_min) * KCAL_PER_HARTREE
        print(f"  {rank+1:<6} micro-cf{idx+1:<9} {e:>14.6f}  {de:>14.3f}")

    # Step 5: create cf1/...cfN/ (top max_conformers only)
    print()
    print("=" * WIDTH)
    print(f"  Step 5: Creating cf1/...cf{cfg['max_conformers']}/ for formal full CSP")
    print(f"  (top {cfg['max_conformers']} of {len(ranked)} by crystal packing energy)")
    print("=" * WIDTH)

    top_ranked = ranked[: cfg["max_conformers"]]
    for rank, (idx, e) in enumerate(top_ranked):
        cf_name = f"cf{rank + 1}"
        cf_dir = os.path.join(run_dir, cf_name)
        os.makedirs(cf_dir, exist_ok=True)
        shutil.copy(conf_paths[idx], os.path.join(cf_dir, "mol.xyz"))
        write_csp_conf(raw_cp, cf_name, os.path.join(cf_dir, "inp.conf"))
        shutil.copy(_DFTB_SETTINGS_SRC, os.path.join(cf_dir, "dftb_settings.json"))
        de = (e - e_min) * KCAL_PER_HARTREE
        print(f"  {cf_name}/  ←  conf_{idx:04d}.xyz  (ΔE = {de:.3f} kcal/mol)")

    print()
    print(f"  {len(top_ranked)} formal CSP folders created.")
    print("=" * WIDTH)


def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def main():
    import time
    t_total = time.time()

    args = parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg, raw_cp = load_config(cfg_path)
    run_dir = os.path.dirname(cfg_path)

    mol_path = os.path.abspath(os.path.join(run_dir, cfg["mol"]))
    if not os.path.exists(mol_path):
        raise FileNotFoundError(f"Molecule file not found: {mol_path}")

    work_dir = os.path.join(run_dir, "_conformer_work")
    conf_dir = os.path.join(run_dir, "conformers")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)

    print("=" * WIDTH)
    print("  Step 1: Conformer Sampling")
    print("=" * WIDTH)
    print(f"  Molecule        : {mol_path}")
    print(f"  Angles/bond     : {cfg['n_angles']} (systematic, {cfg['n_angles']}^N combos)")
    print(f"  Energy window   : {cfg['energy_window']} kcal/mol")
    print(f"  Max conformers  : {cfg['max_conformers']} (uniform sampling)")
    print(f"  Charge          : {cfg['charge']}")
    print()

    n_mini = cfg["max_conformers"] * 2  # 2x for mini-CSP pre-screening
    t1 = time.time()
    surviving_paths, surviving_energies = sample_and_filter(
        mol_path=mol_path,
        n_angles=cfg["n_angles"],
        energy_window=cfg["energy_window"],
        max_conformers=n_mini,
        charge=cfg["charge"],
        work_dir=work_dir,
    )
    t1_end = time.time()

    # ── Write conformers/conf_NNNN.xyz ─────────────────────────────────────
    conf_paths = []
    for i, src in enumerate(surviving_paths):
        dst = os.path.join(conf_dir, f"conf_{i:04d}.xyz")
        shutil.copy(src, dst)
        conf_paths.append(dst)

    # ── Print energy table ─────────────────────────────────────────────────
    e_min = surviving_energies[0]
    print()
    print(f"  Conformers within {cfg['energy_window']:.1f} kcal/mol window: {len(conf_paths)}")
    print()
    print(f"  {'#':<6} {'File':<20} {'ΔE (kcal/mol)':>14}")
    print("  " + "-" * 43)
    for i, (path, e) in enumerate(zip(conf_paths, surviving_energies)):
        rel = (e - e_min) * KCAL_PER_HARTREE
        print(f"  {i:<6} {os.path.basename(path):<20} {rel:>14.3f}")
    print()
    print(f"  Step 1 time: {_fmt_time(t1_end - t1)}")
    print()

    # ── Step 2: Create conformers/micro-cfX/ ──────────────────────────────
    print("=" * WIDTH)
    print("  Step 2: Creating mini-CSP folders")
    print("=" * WIDTH)

    micro_dirs = []
    for i, (conf_path, e) in enumerate(zip(conf_paths, surviving_energies)):
        folder_name = f"micro-cf{i + 1}"
        folder = os.path.join(conf_dir, folder_name)
        os.makedirs(folder, exist_ok=True)
        micro_dirs.append(folder)

        shutil.copy(conf_path, os.path.join(folder, "mol.xyz"))
        write_cf_conf(
            src_cp=raw_cp,
            folder_name=folder_name,
            num_spg=cfg["num_structures_per_spg"],
            out_path=os.path.join(folder, "inp.conf"),
        )

        rel = (e - e_min) * KCAL_PER_HARTREE
        print(f"  conformers/micro-cf{i + 1}/  ←  conf_{i:04d}.xyz  (ΔE = {rel:.3f} kcal/mol)")

    print()
    print(f"  {len(conf_paths)} mini-CSP folders created under conformers/.")

    # ── Step 3: Run gnrs mini-CSP in each micro-cfX/ ──────────────────────
    print()
    print("=" * WIDTH)
    print("  Step 3: Running gnrs mini-CSP")
    print(f"  mpirun -n {cfg['mpi_np']} gnrs -c inp.conf  (per conformer)")
    print("=" * WIDTH)

    t3 = time.time()
    failed = []
    for i, folder in enumerate(micro_dirs):
        folder_name = f"micro-cf{i + 1}"
        log_path = os.path.join(folder, "run.log")
        t_run = time.time()
        print(f"  [{i + 1}/{len(micro_dirs)}] conformers/{folder_name}/ ...", end="", flush=True)

        gnrs_bin = os.path.join(os.path.dirname(__file__), "gnrs_env", "bin", "gnrs")
        cmd = ["mpirun", "-n", str(cfg["mpi_np"]), gnrs_bin, "-c", "inp.conf"]
        env = os.environ.copy()
        for var in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONNOUSERSITE"):
            env.pop(var, None)
        with open(log_path, "w") as log_fh:
            result = subprocess.run(cmd, cwd=folder, env=env, stdout=log_fh, stderr=log_fh)

        elapsed = time.time() - t_run
        if result.returncode == 0:
            print(f"  OK  ({_fmt_time(elapsed)})")
        else:
            print(f"  FAILED (rc={result.returncode}, {_fmt_time(elapsed)})")
            failed.append(folder_name)

    t3_end = time.time()
    print()
    if failed:
        print(f"  WARNING: {len(failed)} run(s) failed: {', '.join(failed)}")
    else:
        print(f"  All {len(micro_dirs)} mini-CSP runs completed successfully.")
    print(f"  Step 3 time: {_fmt_time(t3_end - t3)}")
    print("=" * WIDTH)

    # ── Step 4+5: xTB crystal energies → rank → cf dirs ───────────────────
    print()
    t45 = time.time()
    rank_and_create_cf_dirs(micro_dirs, conf_paths, run_dir, raw_cp, cfg)
    t45_end = time.time()

    print()
    print("=" * WIDTH)
    print("  Timing Summary")
    print("=" * WIDTH)
    print(f"  Step 1  (conformer sampling + xTB SP) : {_fmt_time(t1_end - t1)}")
    print(f"  Step 3  (gnrs mini-CSP, {len(micro_dirs)} conformers) : {_fmt_time(t3_end - t3)}")
    print(f"  Step 4+5 (xTB periodic + cf dirs)     : {_fmt_time(t45_end - t45)}")
    print(f"  Total                                  : {_fmt_time(time.time() - t_total)}")
    print("=" * WIDTH)


def sample_and_filter(
    mol_path, n_angles, energy_window, max_conformers, charge, work_dir,
):
    import numpy as np
    import itertools
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds, rdMolTransforms
    from rdkit.Chem.rdmolfiles import MolToXYZBlock

    xyz_path = os.path.join(work_dir, "input.xyz")
    _to_xyz(mol_path, xyz_path)

    mol = Chem.MolFromXYZFile(xyz_path)
    if mol is None:
        raise RuntimeError(f"RDKit could not read molecule from {xyz_path}")
    rdDetermineBonds.DetermineBonds(mol, charge=charge)

    # Embed one base conformer with MMFF
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    # Find heavy-atom rotatable bonds (exclude pure H-rotation bonds like OH, NH2)
    rot_quartets = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1.0:
            continue
        j = bond.GetBeginAtomIdx(); k = bond.GetEndAtomIdx()
        j_heavy = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()
                   if a.GetIdx() != k and a.GetAtomicNum() != 1]
        k_heavy = [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors()
                   if a.GetIdx() != j and a.GetAtomicNum() != 1]
        if j_heavy and k_heavy:
            rot_quartets.append((j_heavy[0], j, k, k_heavy[0]))

    n_bonds = len(rot_quartets)
    n_total = n_angles ** n_bonds
    step = 360.0 / n_angles
    angle_grid = [i * step for i in range(n_angles)]

    print(f"  Heavy-atom rotatable bonds: {n_bonds}")
    for a, b, c, d in rot_quartets:
        sb = mol.GetAtomWithIdx(b).GetSymbol(); sc = mol.GetAtomWithIdx(c).GetSymbol()
        print(f"    {sb}{b}-{sc}{c}")
    print(f"  Systematic enumeration: {n_angles}^{n_bonds} = {n_total} combinations")
    print(f"  xTB GFN2 single-point (no geometry optimization)...")

    sp_dir = os.path.join(work_dir, "sp")
    os.makedirs(sp_dir, exist_ok=True)

    energies, cand_paths = [], []
    n_failed = 0

    for idx, angle_combo in enumerate(itertools.product(angle_grid, repeat=n_bonds)):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"    [{idx+1}/{n_total}]", flush=True)

        # Clone base conformer and set dihedrals
        mol_copy = Chem.RWMol(mol)
        conf = mol_copy.GetConformer(0)
        for (a, b, c, d), angle in zip(rot_quartets, angle_combo):
            rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, angle)

        conf_work = os.path.join(sp_dir, f"conf_{idx:05d}")
        os.makedirs(conf_work, exist_ok=True)
        xyz_file = os.path.join(conf_work, "mol.xyz")
        with open(xyz_file, "w") as fh:
            fh.write(MolToXYZBlock(mol_copy, confId=0))

        result = subprocess.run(
            ["xtb", "mol.xyz", "--gfn", "2", "--scc",
             "--chrg", str(charge), "--norestart"],
            cwd=conf_work, capture_output=True, text=True,
        )

        if result.returncode != 0:
            n_failed += 1
            continue
        try:
            energy = _parse_xtb_energy(result.stdout)
        except RuntimeError:
            n_failed += 1
            continue

        energies.append(energy)
        cand_paths.append(xyz_file)

    if n_failed:
        print(f"  Warning: {n_failed} xTB single-points failed.")
    if not energies:
        raise RuntimeError("All xTB single-points failed.")

    print(f"  xTB done: {len(cand_paths)} succeeded.")

    # Energy window filter
    paired = sorted(zip(energies, cand_paths), key=lambda x: x[0])
    e_min = paired[0][0]
    window_ha = energy_window / KCAL_PER_HARTREE
    within_window = [(e, p) for e, p in paired if (e - e_min) <= window_ha]
    print(f"  {len(within_window)} candidates within {energy_window:.1f} kcal/mol window.")

    # Uniform sampling across energy range
    if len(within_window) <= max_conformers:
        selected = within_window
    else:
        indices = np.round(np.linspace(0, len(within_window) - 1, max_conformers)).astype(int)
        selected = [within_window[i] for i in indices]
        print(f"  Uniform sampling: {len(selected)} conformers selected.")

    return [p for _, p in selected], [e for e, _ in selected]


def _kabsch_rmsd(pos1, pos2):
    import numpy as np
    p = pos1 - pos1.mean(0)
    q = pos2 - pos2.mean(0)
    H = p.T @ q
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T  # R rotates q → p
    return np.sqrt(((p - q @ R) ** 2).sum() / len(p))


def _rmsd_dedup(sorted_pairs, threshold):
    """Greedy RMSD dedup on sorted (energy, xyz_path) pairs. Keeps lowest-energy first.
    Uses heavy-atom positions only to avoid false positives from freely rotating OH/NH2 H."""
    from ase.io import read as ase_read
    kept = []
    kept_pos = []
    for e, path in sorted_pairs:
        atoms = ase_read(path, parallel=False)
        heavy_mask = atoms.get_atomic_numbers() != 1
        pos = atoms.get_positions()[heavy_mask]
        is_dup = any(_kabsch_rmsd(kp, pos) < threshold for kp in kept_pos)
        if not is_dup:
            kept.append((e, path))
            kept_pos.append(pos)
    return kept


def _to_xyz(src, dst):
    from ase.io import read as ase_read, write as ase_write
    ase_write(dst, ase_read(src), format="xyz")


def _parse_xtb_energy(stdout):
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


if __name__ == "__main__":
    main()
