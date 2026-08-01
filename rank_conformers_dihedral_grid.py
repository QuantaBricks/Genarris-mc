"""
Step 1 of the mini-CSP conformer pipeline: pick which conformers go into CSP.

Usage:
    python rank_conformers_dihedral_grid.py

Edit MOL_PATH/N_KEEP at the top before running for a different molecule/count.
Output: conformer_ranking_grid.json (next to MOL_PATH) -- feed this straight
into run_csp_16conformers.py (its RANKING_JSON points here by default).

Method:
  1. xTB GFN2 geometry-optimize the input molecule once (reference geometry).
  2. For each rotatable bond, sample 3 dihedral angles (0, 120, 240 deg) on
     that reference geometry and enumerate the full combinatorial grid
     (3^n_rotatable_bonds conformers) -- no re-optimization of the grid.
  3. xTB GFN2 single-point energy on each grid conformer, rank by energy,
     keep the lowest N_KEEP.
"""
import os, sys, json, shutil, subprocess, tempfile, itertools
from concurrent.futures import ProcessPoolExecutor

MOL_PATH = "/home/xchen/Test/Genarris/test/lbwl-1/mol.xyz"
N_KEEP = 16
CORE_BUDGET = 16
ANGLES_DEG = [0.0, 120.0, 240.0]

XTB_BIN = "/home/xchen/software/xtb-6.7.1/xtb-dist/bin"
os.environ["PATH"] = XTB_BIN + ":" + os.environ.get("PATH", "")
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, "/home/xchen/Test/Genarris-mc")


def _xtb_optimize_xyz(mol_path):
    work_dir = tempfile.mkdtemp(prefix="xtb_ref_opt_")
    input_xyz = os.path.join(work_dir, "mol.xyz")
    shutil.copy(mol_path, input_xyz)
    cmd = ["xtb", "mol.xyz", "--opt", "--gfn", "2", "--chrg", "0", "--norestart"]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    opt_path = os.path.join(work_dir, "xtbopt.xyz")
    if result.returncode != 0 or not os.path.exists(opt_path):
        raise RuntimeError("reference xTB optimization failed")
    return opt_path


def get_rotatable_bond_dihedrals(mol):
    """Return list of (i, j, k, l) atom-index quadruples, one per rotatable bond."""
    from rdkit import Chem

    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern)
    quads = []
    for j, k in matches:
        atom_j = mol.GetAtomWithIdx(j)
        atom_k = mol.GetAtomWithIdx(k)
        i = next((n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != k), None)
        l = next((n.GetIdx() for n in atom_k.GetNeighbors() if n.GetIdx() != j), None)
        if i is None or l is None:
            continue
        quads.append((i, j, k, l))
    return quads


def _opt_one(args):
    idx, xyz_block = args
    e = None
    work_dir = tempfile.mkdtemp(prefix="xtb_grid_")
    try:
        input_xyz = os.path.join(work_dir, "mol.xyz")
        with open(input_xyz, "w") as f:
            f.write(xyz_block)
        cmd = ["xtb", "mol.xyz", "--sp", "--gfn", "2", "--chrg", "0", "--norestart"]
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
        if result.returncode == 0:
            from gnrs.gnrsutil.conformer import _parse_xtb_energy
            try:
                e = _parse_xtb_energy(result.stdout)
            except RuntimeError:
                e = None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return idx, e, xyz_block


def main():
    from rdkit import Chem
    from rdkit.Chem.rdmolfiles import MolToXYZBlock
    from rdkit.Chem import rdDetermineBonds, rdMolTransforms

    print(f"[1] xTB GFN2 pre-optimizing reference geometry from {MOL_PATH} ...", flush=True)
    opt_xyz = _xtb_optimize_xyz(MOL_PATH)

    mol = Chem.MolFromXYZFile(opt_xyz)
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    quads = get_rotatable_bond_dihedrals(mol)
    n_rot = len(quads)
    n_grid = len(ANGLES_DEG) ** n_rot
    print(f"[2] {n_rot} rotatable bonds -> {len(ANGLES_DEG)}^{n_rot} = {n_grid} "
          f"combinatorial conformers", flush=True)

    conf = mol.GetConformer()
    xyz_blocks = []
    for combo in itertools.product(range(len(ANGLES_DEG)), repeat=n_rot):
        for (i, j, k, l), a_idx in zip(quads, combo):
            rdMolTransforms.SetDihedralDeg(conf, i, j, k, l, ANGLES_DEG[a_idx])
        xyz_blocks.append(MolToXYZBlock(mol))

    print(f"[3] xTB GFN2 single-point on {len(xyz_blocks)} conformers "
          f"({CORE_BUDGET} parallel processes) ...", flush=True)
    results = [None] * len(xyz_blocks)
    with ProcessPoolExecutor(max_workers=CORE_BUDGET) as ex:
        for idx, e, final_xyz in ex.map(_opt_one, list(enumerate(xyz_blocks))):
            results[idx] = (e, final_xyz)
            tag = f"{e:.6f} Ha" if e is not None else "FAILED"
            print(f"    grid_{idx:04d}  {tag}", flush=True)

    succeeded = [(i, e, xyz) for i, (e, xyz) in enumerate(results) if e is not None]
    ranked = sorted(succeeded, key=lambda t: t[1])

    out_dir = os.path.join(os.path.dirname(MOL_PATH), "conformers_grid")
    os.makedirs(out_dir, exist_ok=True)
    kept = []
    for rank_i, (idx, e, xyz) in enumerate(ranked[:N_KEEP]):
        path = os.path.join(out_dir, f"candidate_{rank_i}.xyz")
        with open(path, "w") as f:
            f.write(xyz)
        kept.append({"path": path, "energy_ha": e, "grid_idx": idx})

    print("\n" + "=" * 60)
    print(f"Top-{N_KEEP} lowest-energy conformers (dihedral grid, xTB GFN2)")
    print("=" * 60)
    print(f"{'Rank':>4}  {'Energy (Ha)':>14}  grid_idx")
    print("-" * 60)
    for i, k in enumerate(kept, 1):
        print(f"{i:>4}  {k['energy_ha']:>14.6f}  {k['grid_idx']}")
    print("=" * 60)
    print(f"Total grid points: {len(xyz_blocks)}  succeeded: {len(succeeded)}  kept: {len(kept)}")

    out_json = os.path.join(os.path.dirname(MOL_PATH), "conformer_ranking_grid.json")
    with open(out_json, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"Kept conformers ranking written to {out_json}")


if __name__ == "__main__":
    main()
