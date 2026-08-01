"""
Conformer generation utility: xTB optimization + RDKit ETKDG sampling.

Ranking by crystal energy is handled in StructureGenerationTask via
mini-CSP so that MPI parallelism is properly used.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import os
import shutil
import logging
import subprocess
import tempfile

logger = logging.getLogger("conformer")


def generate_conformations(
    mol_path: str,
    n_candidates: int,
    seed: int = 42,
) -> list[str]:
    """
    Generate candidate conformations from a molecule xyz file.

    Steps:
        1. xTB GFN2 geometry optimization
        2. RDKit ETKDG rotatable-bond sampling

    Args:
        mol_path: Path to input xyz file.
        n_candidates: Number of candidate conformations to generate.
        seed: Random seed.

    Returns:
        List of paths to candidate xyz files.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    from rdkit.Chem.rdmolfiles import MolToXYZBlock

    # Step 1: xTB optimize
    opt_xyz = _xtb_optimize(mol_path)

    # Step 2: Read optimized geometry and determine bonds
    mol = Chem.MolFromXYZFile(opt_xyz)
    if mol is None:
        raise RuntimeError(f"RDKit could not read molecule from {opt_xyz}")
    rdDetermineBonds.DetermineBonds(mol, charge=0)

    n_rot = _count_rotatable_bonds(mol)
    logger.info(f"Rotatable bonds: {n_rot}, requesting {n_candidates} candidates")

    # Step 3: ETKDG sampling with RMSD pruning to avoid near-duplicate conformations
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.ETversion = 2
    params.pruneRmsThresh = 0.5  # discard conformers within 0.5 Å RMSD of each other
    AllChem.EmbedMultipleConfs(mol, numConfs=n_candidates, params=params)

    n_generated = mol.GetNumConformers()
    logger.info(f"ETKDG embedded {n_generated} unique conformations (RMSD threshold 0.5 Å)")

    if n_generated == 0:
        raise RuntimeError("RDKit failed to generate any conformations.")

    # Step 4: Write each conformation to xyz
    out_dir = os.path.join(os.path.dirname(os.path.abspath(mol_path)), "conformers")
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for i in range(n_generated):
        xyz_block = MolToXYZBlock(mol, confId=i)
        path = os.path.join(out_dir, f"candidate_{i}.xyz")
        with open(path, "w") as f:
            f.write(xyz_block)
        paths.append(path)

    logger.info(f"Wrote {len(paths)} candidates to {out_dir}")
    return paths


# ---------------------------------------------------------------------------
# xTB helpers
# ---------------------------------------------------------------------------

def _xtb_optimize(mol_path: str) -> str:
    """Run xTB GFN2 geometry optimization. Falls back to original on failure."""
    work_dir = tempfile.mkdtemp(prefix="xtb_opt_")
    input_xyz = os.path.join(work_dir, "mol.xyz")
    shutil.copy(mol_path, input_xyz)

    cmd = ["xtb", "mol.xyz", "--opt", "--gfn", "2", "--chrg", "0", "--norestart"]
    logger.info(f"xTB optimization: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

    opt_path = os.path.join(work_dir, "xtbopt.xyz")
    if result.returncode != 0 or not os.path.exists(opt_path):
        logger.warning("xTB optimization failed, using original geometry")
        return mol_path

    logger.info("xTB optimization completed")
    return opt_path


def xtb_crystal_singlepoint(atoms, work_dir: str, tag: str = "sp") -> float:
    """
    Run xTB GFN-FF single-point on a periodic crystal structure.

    GFN2 does not support periodic boundary conditions; GFN1 is used
    instead. GFN1 is faster than GFN-FF for medium-sized molecular crystals.

    Args:
        atoms: ASE Atoms object with cell and PBC set.
        work_dir: Working directory for xTB.
        tag: File tag to avoid collisions.

    Returns:
        Total energy in Hartree.
    """
    coord_path = os.path.join(work_dir, f"{tag}_coord")
    _write_xtb_coord(atoms, coord_path)

    cmd = ["xtb", f"{tag}_coord", "--sp", "--gfn", "1", "--chrg", "0", "--norestart"]
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"xTB single-point failed for {tag}:\n{result.stderr[-300:]}"
        )

    return _parse_xtb_energy(result.stdout)


def _write_xtb_coord(atoms, path: str) -> None:
    """Write ASE Atoms as TURBOMOLE coord with $periodic 3 for xTB."""
    ANG2BOHR = 1.8897259886
    lines = ["$coord"]
    for pos, sym in zip(atoms.get_positions(), atoms.get_chemical_symbols()):
        x, y, z = pos * ANG2BOHR
        lines.append(f"   {x:.14f}   {y:.14f}   {z:.14f}   {sym.lower()}")
    lines.append("$periodic 3")
    lines.append("$cell")
    lc = atoms.cell.cellpar()  # (a, b, c, alpha, beta, gamma)
    a, b, c = lc[:3] * ANG2BOHR
    al, be, ga = lc[3:]
    lines.append(
        f"   {a:.10f}   {b:.10f}   {c:.10f}   {al:.6f}   {be:.6f}   {ga:.6f}"
    )
    lines.append("$end")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _parse_xtb_energy(stdout: str) -> float:
    """
    Extract total energy (Hartree) from xTB stdout.
    Returns the last occurrence so optimization output gives the final energy.
    """
    energy = None
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line:
            try:
                energy = float(line.split()[-3])
            except (ValueError, IndexError):
                continue
    if energy is None:
        raise RuntimeError("Could not parse xTB energy from output")
    return energy


def best_crystal_energy_xtb(geometry_out: str, work_dir: str, z: int) -> float:
    """
    Read crystal structures from a geometry.out file and return the minimum
    xTB GFN2 single-point energy per molecule (Ha).

    Skips structures for which xTB fails. Returns inf if no structure succeeds.

    Args:
        geometry_out: Path to the FHI-aims geometry.out produced by cgenarris.
        work_dir: Directory to write temporary POSCAR files and xTB output.
        z: Number of molecules per unit cell (used to divide total energy).

    Returns:
        Minimum energy per molecule in Hartree, or float('inf') on failure.
    """
    from gnrs.parallel.io import str2atoms

    if not os.path.exists(geometry_out):
        return float("inf")

    with open(geometry_out) as f:
        content = f.read()

    blocks = content.split("#######  END  STRUCTURE #######")[:-1]
    if not blocks:
        return float("inf")

    xtb_dir = os.path.join(work_dir, "xtb_sp")
    os.makedirs(xtb_dir, exist_ok=True)

    best_e = float("inf")
    for j, block in enumerate(blocks):
        atoms = str2atoms(block.split("\n"))
        if atoms is None:
            continue
        try:
            e_tot = xtb_crystal_singlepoint(atoms, xtb_dir, tag=f"s{j}")
            e_per_mol = e_tot / z
            if e_per_mol < best_e:
                best_e = e_per_mol
        except Exception as exc:
            logger.warning(f"xTB SP failed for structure {j}: {exc}")

    return best_e


def _count_rotatable_bonds(mol) -> int:
    from rdkit.Chem import rdMolDescriptors
    return rdMolDescriptors.CalcNumRotatableBonds(mol)
