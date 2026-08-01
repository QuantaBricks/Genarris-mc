"""
Step 2 of the mini-CSP conformer pipeline: run CSP end-to-end per conformer.

Usage:
    python run_csp_16conformers.py                          # 16 conformers, full pg_graded
    python run_csp_16conformers.py --n-conformers 4          # fewer conformers
    python run_csp_16conformers.py --mode test               # fast smoke test (10x fewer structures)
    python run_csp_16conformers.py --n-conformers 1 --mode test  # quickest sanity check

Requires conformer_ranking_grid.json to already exist (run
rank_conformers_dihedral_grid.py first). Writes per-conformer results under
WORK_ROOT/conf_N/, prints a final press_energy ranking, and automatically
merges every conformer's dedup structures.json into WORK_ROOT/merged_structures.json.

Fixed end-to-end procedure:
  1. Conformer generation + selection already done (rank_conformers_dihedral_grid.py):
     xTB GFN2 reference optimization -> per-rotatable-bond 3-angle grid ->
     xTB GFN2 single-point per grid conformer -> keep lowest N_KEEP
     (see /home/xchen/Test/Genarris/test/lbwl-1/conformer_ranking_grid.json).
  2. For each kept conformer, run the real production pipeline end-to-end via
     gnrs.cli: generation -> symm_rigid_press -> dedup. Full pg_graded space
     group distribution (all compatible spgs, CSD-frequency-weighted target
     counts), same config validated on conf_4 (~1550 raw -> ~1430 unique,
     73s with 16 MPI ranks). No space-group restriction, no structure-count cap.
  3. Rank the 16 conformers by their best (lowest) press_energy after dedup.

One gnrs.cli job at a time, each using 16 MPI ranks (16-core budget total,
never scaled up to run multiple conformers' jobs concurrently).
"""
import argparse
import os, sys, json, shutil, subprocess, time

# 16 conformers run concurrently, one process each -- cap each process to a
# single thread so BLAS/OpenMP doesn't oversubscribe (16 procs x default
# thread count would massively over-schedule the 16-48 physical cores).
_ENV = os.environ.copy()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _ENV[_v] = "1"

RANKING_JSON = "/home/xchen/Test/Genarris/test/lbwl-1/conformer_ranking_grid.json"
WORK_ROOT = "/home/xchen/Test/Genarris/test/lbwl-1/csp_16conf_grid"
Z = 4
MPI_RANKS = 16  # validated: 16 ranks per conformer job, conformers run
                # sequentially (73s/conformer on conf_4). The 1-process-per
                # conformer / 16-concurrent approach was measured at ~3.3
                # HOURS per conformer for the press step alone -- abandoned.

# "normal": full pg_graded distribution, CSD-frequency-weighted targets
#           tapering from 500 (spg14) down to 1 -- ~1400-1550 raw structures
#           per conformer, ~65-90s/conformer with 16 MPI ranks.
# "test":   pg_graded:0.1 -- same shape scaled 10x down (tapering from 50
#           down to 1) via a runtime scale factor (parsed in C, no
#           recompile needed to change the ratio) -- for fast smoke-testing,
#           ~140-160 raw structures per conformer.
SPG_DIST_BY_MODE = {"normal": "pg_graded", "test": "pg_graded:0.1"}

INP_CONF_TEMPLATE = """[master]
name = {name}
molecule_path = ["mol.xyz"]
z = {z}
log_level = info

[generation]
num_structures_per_spg = 1
sr = 0.95
max_attempts_per_spg = 20000
tol = 0.1
ucv_mean = predict
ucv_mult = 1.5
max_attempts_per_volume = 1000
spg_distribution_type = {spg_dist}
generation_type = crystal
natural_cutoff_mult = 1.1

[symm_rigid_press]
sr = 0.85
method = BFGS
tol = 0.01
natural_cutoff_mult = 1.2
debug_flag = False
maxiter = 5000

[dedup]
stol = 0.5
ltol = 0.5
angle_tol = 10
strictness = normal

[workflow]
tasks = ['generation', 'symm_rigid_press', 'dedup']
"""

CLI_PY = "/home/xchen/Test/Genarris-mc/.venv/bin/python"


def _launch(i, entry, spg_dist):
    conf_dir = os.path.join(WORK_ROOT, f"conf_{i}")
    os.makedirs(conf_dir, exist_ok=True)
    shutil.copy(entry["path"], os.path.join(conf_dir, "mol.xyz"))
    with open(os.path.join(conf_dir, "inp.conf"), "w") as f:
        f.write(INP_CONF_TEMPLATE.format(name=f"conf_{i}", z=Z, spg_dist=spg_dist))
    log_path = os.path.join(conf_dir, "driver_stdout.log")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        ["mpirun", "-n", str(MPI_RANKS), CLI_PY, "-m", "gnrs.cli", "-c", "inp.conf"],
        cwd=conf_dir, stdout=log_f, stderr=subprocess.STDOUT, env=_ENV,
    )
    return conf_dir, log_path, log_f, proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-conformers", type=int, default=16,
                         help="number of top-ranked conformers to run CSP on")
    parser.add_argument("--mode", choices=sorted(SPG_DIST_BY_MODE), default="normal",
                         help="normal: pg_graded (targets taper from 500); "
                              "test: pg_graded_test (same shape, taper from 50)")
    args = parser.parse_args()
    spg_dist = SPG_DIST_BY_MODE[args.mode]

    ranking = json.load(open(RANKING_JSON))
    kept = ranking[:args.n_conformers]
    os.makedirs(WORK_ROOT, exist_ok=True)

    print(f"Running {len(kept)} conformers ONE AT A TIME, "
          f"{MPI_RANKS} MPI ranks each, mode={args.mode} (spg_distribution_type={spg_dist}) ...",
          flush=True)
    t0 = time.time()

    results = []
    for i, entry in enumerate(kept):
        t_start = time.time()
        conf_dir, log_path, log_f, proc = _launch(i, entry, spg_dist)
        proc.wait()
        log_f.close()
        dt = time.time() - t_start
        total_elapsed = time.time() - t0

        dedup_json = os.path.join(conf_dir, "structures", "dedup", "structures.json")
        if proc.returncode != 0 or not os.path.exists(dedup_json):
            print(f"[{i+1}/{len(kept)}] conf_{i}  FAILED (exit={proc.returncode}, "
                  f"this_conf={dt:.1f}s, total={total_elapsed:.1f}s) -- see {log_path}", flush=True)
            results.append((i, entry, None, None, dt))
            continue

        structs = json.load(open(dedup_json))
        n_gen = len(structs)
        best_e, best_name = None, None
        for name, s in structs.items():
            e = s.get("info", {}).get("press_energy")
            if e is not None and (best_e is None or e < best_e):
                best_e, best_name = e, name
        print(f"[{i+1}/{len(kept)}] conf_{i}  DONE  {n_gen} unique structures after dedup, "
              f"best press_energy={best_e}  (this_conf={dt:.1f}s, total={total_elapsed:.1f}s)", flush=True)
        results.append((i, entry, best_e, n_gen, dt))

    print("\n" + "=" * 70)
    print("End-to-end CSP ranking (16 conformers, generation+press+dedup)")
    print("=" * 70)
    ranked = sorted((r for r in results if r[2] is not None), key=lambda r: r[2])
    print(f"{'Rank':>4}  {'conf':>6}  {'press_energy':>14}  {'n_unique':>8}  {'mol_E(Ha)':>12}")
    print("-" * 70)
    for rank_i, (i, entry, best_e, n_gen, dt) in enumerate(ranked, 1):
        print(f"{rank_i:>4}  conf_{i:<3}{best_e:>14.4f}  {n_gen:>8}  {entry['energy_ha']:>12.6f}")
    failed = [r for r in results if r[2] is None]
    if failed:
        print(f"\n{len(failed)} conformer(s) failed: {[f'conf_{r[0]}' for r in failed]}")

    _merge_structures()


def _merge_structures():
    """Always merge every conf_*/structures/dedup/structures.json into one
    file after a run -- do NOT rely on being reminded to do this."""
    merged = {}
    per_conf_counts = {}
    collisions = 0
    conf_dirs = sorted(
        (p for p in os.listdir(WORK_ROOT) if p.startswith("conf_")),
        key=lambda p: int(p.split("_")[1]),
    )
    for conf_id in conf_dirs:
        struct_path = os.path.join(WORK_ROOT, conf_id, "structures", "dedup", "structures.json")
        if not os.path.exists(struct_path):
            continue
        structs = json.load(open(struct_path))
        per_conf_counts[conf_id] = len(structs)
        for key, entry in structs.items():
            merged_key = f"{conf_id}__{key}"
            if merged_key in merged:
                collisions += 1
            entry.setdefault("info", {})
            entry["info"]["conformer_id"] = conf_id
            entry["info"]["conformer_mol_path"] = os.path.join(WORK_ROOT, conf_id, "mol.xyz")
            merged[merged_key] = entry

    out_path = os.path.join(WORK_ROOT, "merged_structures.json")
    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"\nMerged {len(merged)} structures from {len(per_conf_counts)} conformers "
          f"(key collisions: {collisions}) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
