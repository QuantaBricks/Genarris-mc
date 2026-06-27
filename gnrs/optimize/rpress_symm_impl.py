"""
This module implements the RIGID_PRESS algorithm for 
optimizing crystal structures under symmetry constraints.

RIGID_PRESS algorithm is adapted from:
[rigid-press](https://github.com/godotalgorithm/rigid-press)

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom", "Jonathan Moussa"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import numpy as np
import logging

from ase import Atoms
from spglib import get_symmetry_dataset
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger("SymmRigidPress")
# Maximum number of unit cells to sum over before deciding a crystal is too tightly packed
MAX_CELL_SUM = 2000


def standardize_mol(mol: Atoms) -> Atoms:
    """
    Standardize a molecule by moving its center of geometry to origin
    and aligning with principal axes of rotation.

    Args:
        mol: The molecule to standardize

    Returns:
        Standardized molecule
    """
    an = mol.get_atomic_numbers()
    mol.set_atomic_numbers([1 for _ in range(len(mol))])
    com = mol.get_center_of_mass()
    pos = mol.get_positions()
    pos -= com
    mol.set_positions(pos)
    mol.set_atomic_numbers(an)
    return mol


def get_mol_len(ref_mol: Atoms) -> float:
    """
    Calculate the diameter of the smallest sphere that can enclose a molecule.

    Args:
        ref_mol: The reference molecule

    Returns:
        Diameter of the enclosing sphere
    """
    pos = ref_mol.positions
    cog = np.mean(pos, axis=0)
    dist = cdist(pos, [cog])
    return 2 * dist.max()


def is_duplicate(cog1: np.ndarray, cog2: np.ndarray, tol: float = 0.001) -> bool:
    """
    Check if the difference between two centers of geometry is close to an integer.

    Args:
        cog1: First center of geometry
        cog2: Second center of geometry
        tol: Tolerance for comparison

    Returns:
        True if the difference is close to an integer, False otherwise
    """
    intval = np.round(cog1 - cog2)
    is_dup = np.allclose(intval, cog1 - cog2, atol=tol, rtol=0)
    return is_dup


def find_symm(xtal: Atoms, natoms: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Find  unique symmetry operations

    Args:
        xtal: Crystal structure
        natoms: Number of atoms in the molecule

    Returns:
        Tuple of unique rotation matrices, translation vectors, and space group number
    """
    symm = get_symmetry_dataset(
        (xtal.get_cell(), xtal.get_scaled_positions(), xtal.get_atomic_numbers()),
        symprec=1e-3,
    )
    if symm is None:
        # spglib failed to determine the symmetry of this structure
        raise ValueError(
            "spglib get_symmetry_dataset returned None for this crystal "
        )
    spg = symm.number
    rot, trans = symm.rotations, symm.translations

    # Check if the operations are unique
    uniq_rot, uniq_trans = [], []
    pos = xtal.get_scaled_positions(wrap=False)
    asym_frac = pos[:natoms]
    cog_frac = np.mean(asym_frac, axis=0)
    uniq_cog = []

    for i, (r, t) in enumerate(zip(rot, trans)):
        # Find the shift of cog of mol to bring it back to central cell
        new_cog = (r @ cog_frac.T).T + t
        new_cog = np.divmod(new_cog, 1)[1]

        # Check if the pos is unique
        unique = True
        for cog in uniq_cog:
            if is_duplicate(cog, new_cog):
                unique = False
                continue

        if unique:
            uniq_cog.append(new_cog)
            uniq_rot.append(r)
            uniq_trans.append(t)

    if natoms * len(uniq_rot) != len(xtal):
        raise RuntimeError("Unable to find the unique symmetry operations")

    return np.array(uniq_rot), np.array(uniq_trans), spg


def get_lattice_type(spg: int) -> str:
    """
    Get cell type from spg

    Args:
        spg: Space group number

    Returns:
        Cell type
    """
    if spg <= 0 or spg >= 230:
        return "invalid"
    if spg <= 2:
        return "triclinic"
    elif spg <= 15:
        return "monoclinic"
    elif spg <= 74:
        return "orthorhombic"
    elif spg <= 142:
        return "tetragonal"
    elif spg <= 167:
        return "trigonal"
    elif spg <= 194:
        return "hexagonal"
    else:
        return "cubic"


class RigidPressSymm:
    """
    Class to optimize a crystal structure under symmetry constraints
    using the RIGID_PRESS algorithm.

    This class implements the constrained optimization of crystal structures
    while preserving the space group symmetry.
    """

    def __init__(
        self,
        mol: Atoms,
        xtal: Atoms,
        radius: np.ndarray,
        int_scale: float = 0.1,
        method: str = "BFGS",
        tol: float = 0.01,
        maxiter: int = 5000,
        vol_tol: float = 10,
        debug_flag: bool = False,
        rank: int = 0,
    ):
        """
        Initialize the optimization class.

        Args:
            mol: Reference molecule
            xtal: Initial crystal structure
            radius: Interaction radius matrix
            int_scale: Interaction energy scaling factor
            method: Optimization method for scipy.optimize.minimize
            tol: Convergence tolerance for optimization
            maxiter: Maximum number of iterations
            vol_tol: Minimum allowed volume in callback function
            debug_flag: Whether to print debug information
            rank: Process rank for parallel computation
        """
        self.logger = logger
        self.int_scale = int_scale
        self.method = method
        self.tol = tol
        self.vol_tol = vol_tol
        self.maxiter = maxiter
        self.debug_flag = debug_flag
        self.rank = rank

        # Get mol properties
        self.ref_mol = standardize_mol(mol)
        self.mol_length = get_mol_len(self.ref_mol)
        self.natoms = len(self.ref_mol)

        # Get unique symmetry operations
        self.symm_rot, self.symm_trans, self.spg = find_symm(xtal, self.natoms)
        self.nmol = len(self.symm_rot)
        xtal.spg = self.spg

        self.radius = radius[0 : self.natoms, 0 : self.natoms].ravel()
        # Interaction distance = mol length + 2x max cutoff
        self.D = self.mol_length + 2 * np.max(self.radius)
        self.an = xtal.get_atomic_numbers().tolist()
        self._last_energy = np.inf

    def find_pairs(self, xtal: Atoms) -> dict:
        """
        Find the molecule pairs that are within the interaction distance of the central cell.

        Args:
            xtal: ASE Atoms object

        Returns:
            A dict with key as mol indices pair and value as lattice
            displacements that are within interaction distance.
        """
        # Calculate centres of geometry of each molecule in fractional and cartesian
        pos = xtal.get_scaled_positions(wrap=False)
        lattice = xtal.cell.array
        cog_f = np.array(
            [
                np.mean(pos[i * self.natoms : (i + 1) * self.natoms], axis=0)
                for i in range(self.nmol)
            ]
        )
        cog_c = cog_f @ lattice

        # The bounds of the super cell considered
        cell_lengths = xtal.cell.lengths()
        max_lat = np.ceil(self.D / cell_lengths + 2).astype(int)
        min_lat = -max_lat

        # Test if a crystal is too packed to continue
        if np.prod(max_lat) > MAX_CELL_SUM or np.any(max_lat <= 0):
            return {"inf": True}

        # Create combinations of integer displacements in fractional space
        lat_ranges = [np.arange(min_lat[i], max_lat[i]) for i in range(3)]
        comb_lat = np.array(np.meshgrid(*lat_ranges)).T.reshape(-1, 3)

        # Calculate possible displacements in real space
        lattice_dis = comb_lat @ lattice

        interact_pairs = {}
        # Cross pairs within range, (0,0,0) are molecules within the cell
        for mol1 in range(self.nmol):
            for mol2 in range(mol1 + 1, self.nmol):
                disp = cog_c[mol1] - (cog_c[mol2] + lattice_dis)
                dist = np.linalg.norm(disp, axis=1)
                interact_pairs[(mol1, mol2)] = comb_lat[dist < self.D]

        # Periodic pairs within range
        # Molecule will interact with its own mirror image under periodic boundaries
        # (0,0,0) is the same molecule hence excluded
        comb_lat_non_zero = comb_lat[~np.all(comb_lat == 0, axis=1)]
        non_zero_disp = comb_lat_non_zero @ lattice
        dist = np.linalg.norm(non_zero_disp, axis=1)
        for mol1 in range(self.nmol):
            interact_pairs[(mol1, mol1)] = comb_lat_non_zero[dist < self.D]
        return interact_pairs

    def interaction_kernel(
        self, dist: np.ndarray, radius: np.ndarray, weight: float
    ) -> float:
        """
        Define how atoms of different molecules interact.

        Args:
            dist: Array of distances between atoms
            radius: Array of cutoff radii
            weight: Scaling factor for interaction energy

        Returns:
            Total interaction energy
        """
        energy = np.zeros(len(dist))
        energy[dist < radius] = np.inf
        energy[dist > self.D] = 0
        energy[(dist > radius) & (dist < self.D)] = (
            weight * (self.D - dist) / (dist - radius)
        )[(dist > radius) & (dist < self.D)]
        return energy.sum()

    def _kernel_energy_and_dEdr(
        self, dist_flat: np.ndarray, radius_flat: np.ndarray, weight: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return per-pair (energy, dE/dr) arrays for the repulsive kernel.

        The kernel is  weight*(D-r)/(r-sigma) for sigma < r < D, inf for r <= sigma.
        Its derivative w.r.t. distance: -weight*(D-sigma)/(r-sigma)^2.
        """
        e = np.zeros(len(dist_flat))
        dedr = np.zeros(len(dist_flat))
        inf_mask = dist_flat < radius_flat
        mid_mask = (dist_flat > radius_flat) & (dist_flat < self.D)
        e[inf_mask] = np.inf
        d = dist_flat[mid_mask]
        r = radius_flat[mid_mask]
        gap = d - r
        e[mid_mask] = weight * (self.D - d) / gap
        dedr[mid_mask] = -weight * (self.D - r) / gap ** 2
        return e, dedr

    def pair_energy_and_forces(
        self, xtal: Atoms, mol1: int, mol2: int, weight: float, interact_pairs: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute interaction energy, Cartesian atomic forces, and lattice stress tensor
        for one molecule pair.

        Forces are accumulated into a zeroed array of shape (n_atoms_crystal, 3).
        The lattice stress T[c,b] = Σ_{p,i,j} (-dedr)*r̂[b]*p[c] captures the direct
        dependence of pair energy on the cell matrix L through periodic displacements
        p@L.  It must be contracted with dL/d(state) to complete the gradient.

        Returns (np.inf, zeros, zeros) if any atom overlap is detected.
        """
        na = self.natoms
        lattice = xtal.cell.array
        pairs = interact_pairs[(mol1, mol2)]
        forces = np.zeros_like(xtal.positions)
        T = np.zeros((3, 3))
        if len(pairs) == 0:
            return 0.0, forces, T

        pos1 = xtal.positions[mol1 * na : (mol1 + 1) * na]  # (na, 3)
        pos2 = xtal.positions[mol2 * na : (mol2 + 1) * na]  # (na, 3)

        lattice_disps = pairs @ lattice  # (np, 3)
        np_ = len(lattice_disps)

        # Stack all displaced copies of mol2: (np*na, 3)
        pos2_all = pos2[np.newaxis, :, :] + lattice_disps[:, np.newaxis, :]
        pos2_flat = pos2_all.reshape(-1, 3)

        dist = cdist(pos1, pos2_flat)  # (na, np*na)
        radius_2d = np.tile(self.radius.reshape(na, na), (1, np_))  # (na, np*na)

        e_arr, dedr_arr = self._kernel_energy_and_dEdr(
            dist.ravel(), radius_2d.ravel(), weight
        )
        if np.any(np.isinf(e_arr)):
            return np.inf, forces, T

        total_e = e_arr.sum()

        # dr[i, kj, :] = pos1[i] - pos2_flat[kj]  shape: (na, np*na, 3)
        dr = pos1[:, np.newaxis, :] - pos2_flat[np.newaxis, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_hat = np.where(
                dist[:, :, np.newaxis] > 0, dr / dist[:, :, np.newaxis], 0.0
            )  # (na, np*na, 3)

        dedr_2d = dedr_arr.reshape(na, np_ * na)
        dedr_3d = dedr_2d.reshape(na, np_, na)          # (na_1, np, na_2)
        neg_dedr_3d = -dedr_3d                          # (na_1, np, na_2)
        r_hat_4d = r_hat.reshape(na, np_, na, 3)        # (na_1, np, na_2, 3)

        # dE/d(pos1[i]) = Σ_{kj} dedr[i,kj] * r_hat[i,kj]  → force: subtract
        grad_mol1 = np.einsum("ij,ijk->ik", dedr_2d, r_hat)  # (na, 3)
        forces[mol1 * na : (mol1 + 1) * na] -= grad_mol1

        # dE/d(pos2[j]) = Σ_{i,k} dedr[i,kj] * r_hat[i,kj]  → force: add
        # (dedr < 0 and r̂ points from mol2 to mol1, so force pushes mol2 away)
        grad_mol2 = np.einsum("ikj,ikjl->jl", dedr_3d, r_hat_4d)  # (na_2, 3)
        forces[mol2 * na : (mol2 + 1) * na] += grad_mol2

        # Lattice stress: T[c,b] = Σ_{p,i,j} (-dedr)*r̂[b]*pairs[p,c]
        # R_p[p,b] = Σ_{i,j} (-dedr[i,p,j]) * r̂[i,p,j,b]  shape: (np, 3)
        R_p = np.einsum("ikj,ikjl->kl", neg_dedr_3d, r_hat_4d)
        T = pairs.T @ R_p  # (3, np) @ (np, 3) = (3, 3);  T[c,b]

        return total_e, forces, T

    def total_energy_and_grad(self, state: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute total energy and gradient w.r.t. the state vector.

        Returns (energy, grad) so it can be passed as ``fun`` with ``jac=True``
        to scipy.optimize.minimize.

        Analytical atomic forces are computed from pair interactions; the
        gradient w.r.t. the reduced state vector is obtained by contracting
        those forces with the position Jacobian d(cart_positions)/d(state),
        which is evaluated cheaply via finite differences of create_xtal()
        (no pair-distance computations involved).
        """
        state_copy = np.array(state, dtype=float)
        state_std = self.standardize_state(state_copy)
        xtal = self.create_xtal(state_std)

        interact_pairs = self.find_pairs(xtal)
        if "inf" in interact_pairs:
            self._last_energy = np.inf
            return np.inf, np.zeros(len(state))

        forces = np.zeros_like(xtal.positions)
        T_total = np.zeros((3, 3))
        energy = 0.0
        w_pair = self.int_scale / (self.natoms * self.natoms)

        for key in interact_pairs:
            mol1, mol2 = key
            e, f, T = self.pair_energy_and_forces(xtal, mol1, mol2, w_pair, interact_pairs)
            if np.isinf(e):
                self._last_energy = np.inf
                return np.inf, np.zeros(len(state))
            energy += e
            forces += f
            T_total += T

        V = xtal.get_volume()
        energy += V
        self._last_energy = energy

        # Cheap finite-difference Jacobian: d(cart_positions)/d(state_k) and
        # d(cell_matrix)/d(state_k).  create_xtal() is pure geometry so this is fast.
        pos0 = xtal.positions.ravel()
        L0 = xtal.cell.array
        n_state = len(state_std)
        J = np.zeros((len(pos0), n_state))
        dV_dstate = np.zeros(n_state)
        dL_dstate = np.zeros((3, 3, n_state))
        eps = 1e-6
        for k in range(n_state):
            s_p = state_std.copy()
            s_p[k] += eps
            xtal_p = self.create_xtal(s_p)
            J[:, k] = (xtal_p.positions.ravel() - pos0) / eps
            dV_dstate[k] = (xtal_p.get_volume() - V) / eps
            dL_dstate[:, :, k] = (xtal_p.cell.array - L0) / eps

        # Full gradient:
        #   dE/d(state[k]) = J^T @ (-forces)          [atom-position contribution]
        #                   + T_total : dL/d(state[k]) [periodic-displacement contribution]
        #                   + dV/d(state[k])           [volume contribution]
        lat_stress = np.einsum("cb,cbk->k", T_total, dL_dstate)
        grad_state = J.T @ (-forces.ravel()) + lat_stress + dV_dstate
        return energy, grad_state

    def pair_energy(
        self, xtal: Atoms, mol1: int, mol2: int, weight: float, interact_pairs: dict
    ) -> float:
        """
        Compute the interaction energy between a pair of molecules.

        Args:
            xtal: ASE Atoms object
            mol1: Index of first molecule
            mol2: Index of second molecule
            weight: Scaling factor for interaction energy
            interact_pairs: Dictionary of interacting pairs and their lattice displacements

        Returns:
            Total energy of the pair
        """
        lattice = xtal.cell.array
        pairs = interact_pairs[(mol1, mol2)]
        pos1 = xtal.positions[mol1 * self.natoms : (mol1 + 1) * self.natoms]
        # Combine all possible lattice displacements of mol2 into a single vector
        pos2 = (
            xtal.positions[mol2 * self.natoms : (mol2 + 1) * self.natoms][
                :, :, np.newaxis
            ]
            + (lattice.T @ pairs.T)[np.newaxis, :, :]
        )
        pos2 = pos2.transpose(2, 0, 1).reshape(-1, 3)
        dist = cdist(pos1, pos2).T.reshape(-1)
        stretched_radius = np.resize(self.radius, dist.shape)
        return self.interaction_kernel(dist, stretched_radius, weight)

    def total_energy(self, state: np.ndarray) -> float:
        """
        Calculate the total energy of the crystal structure.

        The total energy is the sum of:
        1. Interaction energy between molecules
        2. Unit cell volume

        Args:
            state: State vector containing lattice parameters
            and molecular position/orientation

        Returns:
            Total energy (sum of volume and interaction energy)
        """
        # Convert state vector to a standardized form
        state_copy = np.array(state, dtype=float)
        state_std = self.standardize_state(state_copy)

        # Create crystal structure from state vector
        xtal = self.create_xtal(state_std)

        # Find interacting molecule pairs
        interact_pairs = self.find_pairs(xtal)
        if "inf" in interact_pairs:
            return np.inf

        # Calculate interaction energy
        energy = 0.0
        w_pair = self.int_scale / (self.natoms * self.natoms)

        for key in interact_pairs:
            mol1, mol2 = key
            energy += self.pair_energy(xtal, mol1, mol2, w_pair, interact_pairs)

        return energy + xtal.get_volume()

    def generate_pos(
        self, asym: np.ndarray, lattice: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        Generate full crystal coordinates from asymmetric unit and symmetry operations

        Args:
            asym: Asymmetric unit coordinates
            lattice: Lattice parameters

        Returns:
            Tuple of full crystal coordinates and number of symmetry operations
        """

        rot, trans = self.symm_rot, self.symm_trans
        pos = np.zeros((len(asym) * len(rot), 3))
        asym_frac = np.linalg.solve(lattice.T, asym.T).T
        cog_frac = np.mean(asym_frac, axis=0)

        for i, (r, t) in enumerate(zip(rot, trans)):
            # Find the shift of cog of mol from the central cell
            new_cog = (r @ cog_frac.T).T + t
            shift = np.divmod(new_cog, 1)[1] - new_cog

            unshifted_pos = (r @ asym_frac.T).T + t
            pos[i * self.natoms : (i + 1) * self.natoms, :] = (
                unshifted_pos + shift
            )  # Move inside central cell

        return pos, len(rot)

    def create_xtal(self, state: np.ndarray) -> Atoms:
        """
        Create an ASE Atoms object from the state vector

        Args:
            state: State vector

        Returns:
            ASE Atoms object
        """

        lattice_type = get_lattice_type(self.spg)

        if lattice_type == "triclinic":
            lattice = [
                [state[0], 0, 0],
                [state[1], state[2], 0],
                [state[3], state[4], state[5]],
            ]

        elif lattice_type == "monoclinic":
            lattice = [[state[0], 0, 0], [0, state[1], 0], [state[2], 0, state[3]]]

        elif lattice_type == "orthorhombic":
            lattice = [[state[0], 0, 0], [0, state[1], 0], [0, 0, state[2]]]

        elif lattice_type == "tetragonal":
            lattice = [[state[0], 0, 0], [0, state[0], 0], [0, 0, state[1]]]

        elif lattice_type in ["hexagonal", "trigonal"]:
            gamma = 2 * np.pi / 3  # in radians
            lattice = [
                [state[0], 0, 0],
                [state[0] * np.cos(gamma), state[0] * np.sin(gamma), 0],
                [0, 0, state[1]],
            ]

        elif lattice_type == "cubic":
            lattice = [[state[0], 0, 0], [0, state[0], 0], [0, 0, state[0]]]

        else:
            raise Exception("Invalid spg")

        lattice = np.array(lattice)
        asym_cog = state[-6:-3]
        asym_angles = state[-3:]

        # create asymm unit from mol position and angles
        rot = Rotation.from_euler("ZYX", asym_angles)
        asym = rot.apply(self.ref_mol.positions)
        asym += asym_cog

        # generate full crystal using symmetry
        pos, nmol = self.generate_pos(asym, lattice)
        xtal = Atoms(cell=lattice, scaled_positions=pos, pbc=True)
        return xtal

    def create_state(self, xtal: Atoms) -> np.ndarray:
        """
        Create an optimization state vector from ASE Atoms object.
        State vector is the lattice vector + position of asymm molecule
        + orientation of asymmetric molecule

        Args:
            xtal: ASE Atoms object

        Returns:
            State vector
        """

        lattice = xtal.cell.array
        asym = xtal.positions[0 : len(self.ref_mol)]
        asym_cog = np.mean(asym, axis=0)
        # Move to asym unit to the origin
        asym_shifted = asym - asym_cog

        # Compute the orientation of asym molecule using Kabsch method
        R, rmsd = Rotation.align_vectors(asym_shifted, self.ref_mol.positions)
        if (rmsd) > 0.1:
            raise Exception("Can not create rotation")
        asym_angles = R.as_euler("ZYX")

        # Create state vector with independent lattice params
        lattice_type = get_lattice_type(self.spg)
        # All six lattice params can vary
        if lattice_type == "triclinic":
            state = [
                lattice[0][0],
                lattice[1][0],
                lattice[1][1],
                lattice[2][0],
                lattice[2][1],
                lattice[2][2],
            ]

        elif lattice_type == "monoclinic":
            state = [lattice[0][0], lattice[1][1], lattice[2][0], lattice[2][2]]

        elif lattice_type == "orthorhombic":
            state = [lattice[0][0], lattice[1][1], lattice[2][2]]

        elif lattice_type in ["tetragonal", "hexagonal", "trigonal"]:
            state = [lattice[0][0], lattice[2][2]]

        elif lattice_type == "cubic":
            state = [lattice[0][0]]

        else:
            raise Exception("Invalid spg")

        state += asym_cog.tolist()
        state += asym_angles.tolist()

        return state

    def standardize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Standardize the state vector

        Args:
            state: State vector

        Returns:
            Standardized state vector
        """
        lattice_type = get_lattice_type(self.spg)
        if lattice_type == "triclinic":
            q = int(state[4] / state[2])
            state[3] = state[3] - q * state[1]
            state[4] = state[4] - q * state[2]

            q = int(state[1] / state[0])
            state[1] = state[1] - q * state[0]

            q = int(state[3] / state[0])
            state[3] = state[3] - q * state[0]

        if lattice_type == "monoclinic":
            q = int(state[2] / state[0])
            if abs(q) > 2:
                state[2] = state[2] - q * state[0]

        return state

    def callback(self, xk: np.ndarray) -> None:
        """
        Callback function for the optimizer

        Args:
            xk: Current state vector
        """
        if self._last_energy < self.vol_tol:
            if self.debug_flag:
                self.logger.debug(f"Failed optimization in {self.rank}")
            raise StopIteration("Custom stopping condition met")
        return self.standardize_state(xk)

    def objective_function(self, state: np.ndarray) -> float:
        """
        Function to be minimized.

        Args:
            state: State vector

        Returns:
            Total energy
        """

        return self.total_energy(state)

    def run(self, xtal: Atoms) -> bool:
        """
        Minimize the energy using Scipy local optimizer.

        Args:
            xtal: Initial crystal structure

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Create initial state vector
            state = self.create_state(xtal)
            state_std = self.standardize_state(state)

            # Check if initial state is valid
            energy, _ = self.total_energy_and_grad(state_std)
            if np.isinf(energy):
                return False

            # Run optimization with analytical gradient (jac=True: fun returns (f, grad))
            res = minimize(
                self.total_energy_and_grad,
                state,
                method=self.method,
                jac=True,
                tol=self.tol,
                options={"disp": self.debug_flag, "maxiter": self.maxiter},
                callback=self.callback,
            )

            # Update structure with optimized parameters
            xtal_opt = self.create_xtal(res.x)
            xtal.set_cell(xtal_opt.cell)
            xtal.set_positions(xtal_opt.positions)
            if res.success:
                xtal.info['press_energy'] = float(res.fun)
            return res.success
        except (StopIteration, RuntimeError):
            return False
