"""
Duplicate structure removal using pymatgen StructureMatcher.

Structures are grouped by space group for computational efficiency,
then within each space group a reference structure is broadcast to all MPI ranks
and compared against the remaining candidates in parallel.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import itertools
import logging
import math
import random
from collections import defaultdict

import numpy as np
from mpi4py import MPI
from ase.atoms import Atoms
from scipy.spatial.transform import Rotation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

import gnrs.parallel as gp
import gnrs.output as gout

_TAG_WORK = 50
_TAG_RESULT = 51
_TAG_SHUTDOWN = 52

logger = logging.getLogger("dedup")


def group_by_spg(structs: dict[str, Atoms]) -> dict[int, dict[str, Atoms]]:
    """
    Group structures by space group.

    Args:
        structs: {name: Atoms}.

    Returns:
        {spg: {name: Atoms, ...}}.
    """
    groups: dict[int, dict[str, Atoms]] = defaultdict(dict)
    for name, xtal in structs.items():
        spg = xtal.info.get("spg")
        groups[spg][name] = xtal
    return groups


def group_by_volume(
    structs: dict[str, Atoms],
    vol_tol: float = 0.05,
    n_buckets: int = 10,
) -> list[dict[str, Atoms]]:
    """
    Sub-group structures by unit cell volume into up to n_buckets bins with
    roughly EQUAL STRUCTURE COUNTS (quantile bins by sorted volume rank, not
    equal-width volume ranges) -- dedup_bucket is O(bucket_size^2) pairwise
    StructureMatcher comparisons, so an equal-width split is worthless
    whenever volumes cluster unevenly (e.g. most structures within a narrow
    range and a few outliers stretching it): one bucket ends up holding most
    of the pool while the rest sit nearly empty, and the O(n^2) cost is
    dominated entirely by that one oversized bucket. Equal-count binning
    bounds the worst-case bucket size to about len(structs)/n_buckets.
    Adjacent buckets share a small overlap so duplicates straddling a
    boundary are compared in both buckets.

    Args:
        structs: {name: Atoms}.
        vol_tol: Relative overlap half-width at bucket boundaries.
        n_buckets: Number of buckets to split into (default 10).

    Returns:
        List of sub-group dicts (with overlap).
    """
    if not structs:
        return []

    items = sorted(structs.items(), key=lambda kv: kv[1].get_volume())
    vols = [xtal.get_volume() for _, xtal in items]
    v_min, v_max = vols[0], vols[-1]

    if v_min == v_max:
        return [dict(items)]

    n = min(n_buckets, len(items))
    # Equal-count boundaries: split sorted items into n contiguous chunks of
    # (roughly) equal size, then read off the volume at each chunk boundary
    # to get overlap-comparison edges.
    chunk_bounds = [round(i * len(items) / n) for i in range(n + 1)]
    edges = [v_min - 1e-6] + [vols[chunk_bounds[k]] for k in range(1, n)] + [v_max + 1e-6]

    buckets: list[dict[str, Atoms]] = [
        dict(items[chunk_bounds[k]:chunk_bounds[k + 1]]) for k in range(n)
    ]

    # Overlap: copy boundary structures into the adjacent bucket so duplicates
    # straddling a boundary edge are still compared in both buckets.
    overlap_width = vol_tol * (v_max - v_min) / n
    for i in range(n - 1):
        boundary = edges[i + 1]
        for name, xtal in list(buckets[i].items()):
            if boundary - xtal.get_volume() <= overlap_width:
                buckets[i + 1][name] = xtal
        for name, xtal in list(buckets[i + 1].items()):
            if xtal.get_volume() - boundary <= overlap_width:
                buckets[i][name] = xtal

    return [b for b in buckets if b]


def _select(
    candidates: dict[str, Atoms],
    energy_key: str | None
) -> str:
    """
    Select one structure from a set of duplicates.

    If energy_key is provided, the lowest-energy structure is chosen. 
    Otherwise a random one is chosen.

    Args:
        candidates: {name: Atoms} duplicates.
        energy_key: Key in Atoms.info for energy, or None.

    Returns:
        Name of the chosen structure.
    """
    if energy_key is not None:
        energies = []
        for name, xtal in candidates.items():
            e = xtal.info.get(energy_key)
            if e is not None:
                energies.append((name, float(e)))
        if len(energies) == len(candidates):
            return min(energies, key=lambda x: x[1])[0]

    return random.choice(sorted(candidates.keys()))

def _scatter_structs(pool: dict[str, Atoms]) -> dict[str, Atoms]:
    """
    Master scatters a dict of structures evenly across ranks.
    """
    scatter_list = None
    if gp.is_master:
        items = list(pool.items())
        n = len(items)
        per_rank = n // gp.size
        remainder = n % gp.size
        scatter_list = []
        start = 0
        for r in range(gp.size):
            chunk = per_rank + (1 if r < remainder else 0)
            scatter_list.append(dict(items[start : start + chunk]))
            start += chunk
    return gp.comm.scatter(scatter_list, root=0)


def dedup_parallel(
    spg_groups: dict[int, dict[str, Atoms]],
    matcher: StructureMatcher,
    energy_key: str | None,
    ref_mol_positions: np.ndarray | None = None,
    natoms: int | None = None,
    nmol: int | None = None,
    strictness: str = "normal",
) -> dict[str, Atoms]:
    """
    Dispatch one spg pool at a time to worker ranks (rank 0 = dispatcher).
    Each worker receives one spg pool, splits it into volume buckets,
    and deduplicates each bucket sequentially — no MPI inside.

    Args:
        spg_groups: {spg: {name: Atoms}} (only meaningful on rank 0).
        matcher: StructureMatcher instance (used only if ref_mol_positions is
            None, i.e. the fast fingerprint path is unavailable).
        energy_key: Energy key for selecting best duplicate.
        ref_mol_positions: standardized reference molecule positions
            (natoms, 3), enabling the fast fingerprint-based dedup path (see
            fast_dedup_bucket). If None, falls back to the slow generic
            pymatgen StructureMatcher path (dedup_bucket).
        natoms: atoms per molecule (required with ref_mol_positions).
        nmol: molecules per unit cell (required with ref_mol_positions).
        strictness: one of "loose", "normal", "rigid" (see DEDUP_PRESETS).
            Only affects the fast fingerprint path.

    Returns:
        Combined unique structures (broadcast to all ranks).
    """
    use_fast = ref_mol_positions is not None
    preset = resolve_dedup_preset(strictness)

    def _dedup_bucket(bucket):
        if use_fast:
            return fast_dedup_bucket(
                bucket, ref_mol_positions, natoms, nmol, energy_key,
                energy_tol=preset["energy_tol"],
                pos_tol_frac=preset["pos_tol_frac"],
                angle_tol_deg=preset["angle_tol_deg"],
                cellpar_len_rtol=preset["len_rtol"],
                cellpar_angle_tol=preset["cellpar_angle_tol"],
            )
        return dedup_bucket(bucket, matcher, energy_key)

    if gp.is_master:
        queue = list(spg_groups.items())  # [(spg, pool), ...]
        kept = {}
        active = gp.size - 1

        total = len(queue)
        done = 0

        if gp.size == 1:
            # No worker ranks to dispatch to (singleton/1-rank run): the
            # master must do the dedup work itself instead of sending to a
            # nonexistent worker, or every spg pool is silently dropped.
            for spg, pool in queue:
                for bucket in group_by_volume(pool):
                    kept.update(_dedup_bucket(bucket))
                done += 1
                gout.emit(f"Dedup: {done}/{total} spgs done, {len(kept)} unique so far")
            return kept

        for worker in range(1, gp.size):
            if queue:
                gp.comm.send(queue.pop(0), dest=worker, tag=_TAG_WORK)
            else:
                gp.comm.send(None, dest=worker, tag=_TAG_SHUTDOWN)
                active -= 1

        while active > 0:
            status = MPI.Status()
            result = gp.comm.recv(source=MPI.ANY_SOURCE, tag=_TAG_RESULT, status=status)
            kept.update(result)
            done += 1
            gout.emit(f"Dedup: {done}/{total} spgs done, {len(kept)} unique so far")
            worker = status.Get_source()
            if queue:
                gp.comm.send(queue.pop(0), dest=worker, tag=_TAG_WORK)
            else:
                gp.comm.send(None, dest=worker, tag=_TAG_SHUTDOWN)
                active -= 1

        return gp.comm.bcast(kept, root=0)

    else:
        kept = {}
        while True:
            status = MPI.Status()
            item = gp.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == _TAG_SHUTDOWN:
                break
            spg, pool = item
            for bucket in group_by_volume(pool):
                kept.update(_dedup_bucket(bucket))
            gp.comm.send(kept, dest=0, tag=_TAG_RESULT)
            kept = {}

        return gp.comm.bcast(None, root=0)


def dedup_bucket(
    bucket: dict[str, Atoms],
    matcher: StructureMatcher,
    energy_key: str | None,
    energy_tol: float = 0.01,
) -> dict[str, Atoms]:
    """
    Deduplicate a single volume bucket on one rank, no MPI.

    Args:
        bucket: {name: Atoms} structures in this volume bucket.
        matcher: Configured StructureMatcher instance.
        energy_key: Key in Atoms.info for energy, or None.
        energy_tol: If both structures have finite energies differing by more
            than this (eV), skip StructureMatcher and treat as non-duplicate.

    Returns:
        {name: Atoms} unique structures.
    """
    import math

    pool = dict(bucket)
    kept = {}
    while pool:
        ref_name, ref_xtal = next(iter(pool.items()))
        pool.pop(ref_name)
        pmg_ref = AseAtomsAdaptor.get_structure(ref_xtal)
        e_ref = ref_xtal.info.get(energy_key) if energy_key else None
        if e_ref is not None:
            e_ref = float(e_ref)
            if math.isinf(e_ref):
                e_ref = None
        cluster = {ref_name: ref_xtal}
        for name in list(pool.keys()):
            if e_ref is not None:
                e = pool[name].info.get(energy_key)
                if e is not None:
                    e = float(e)
                    if not math.isinf(e) and abs(e - e_ref) > energy_tol:
                        continue
            pmg_xtal = AseAtomsAdaptor.get_structure(pool[name])
            if matcher.fit(pmg_ref, pmg_xtal):
                cluster[name] = pool.pop(name)
        best = _select(cluster, energy_key)
        kept[best] = cluster[best]
    return kept


# ---------------------------------------------------------------------------
# Fast fingerprint-based dedup, specific to RigidPressSymm-optimized
# structures.
#
# Every optimized structure is generated from ONE asymmetric-unit molecule
# (center of geometry + orientation) plus a small set of KNOWN symmetry
# operations (see gnrs.optimize.rpress_symm_impl.RigidPressSymm), so within a
# given spg group, two structures are the same physical packing iff their
# molecule centers-of-geometry + orientations coincide (up to the ambiguity
# of which molecule is picked as "the" asymmetric unit). Comparing that
# directly (nmol points, typically <=8) is ~1000x+ faster than asking a
# generic pymatgen StructureMatcher to rediscover the symmetry relationship
# from the full atomic structure (nmol*natoms atoms) from scratch.
#
# Validated against pymatgen's StructureMatcher on real generated/optimized
# structures: ~98% exact agreement (46/47 vs 46 kept on one 57-structure
# test bucket), all pymatgen-kept structures were a subset of the fast
# method's kept set (i.e. the one disagreement was the fast method being
# slightly more conservative -- an extra near-duplicate kept, not two
# genuinely different structures wrongly merged). ~1700x faster.
#
# Strictness presets. "normal" is deliberately a bit tighter than the
# project's original pymatgen-based defaults (ltol=0.5, stol=0.5,
# angle_tol=10, no energy pre-filter cutoff), which were tuned for a
# supercell/basis-aware matcher, not this direct cellpar+cog comparison.
# "loose" reproduces roughly that original looseness; "rigid" is for callers
# who would rather under-merge than risk collapsing two distinct packings.
#   len_rtol:          relative tolerance on cell lengths a/b/c (~ltol).
#   cellpar_angle_tol:  absolute tolerance (deg) on cell angles alpha/beta/gamma.
#   pos_tol_frac:      fractional-coordinate tolerance for molecule COG match (~stol).
#   angle_tol_deg:     tolerance (deg) on relative molecule orientation.
#   energy_tol:        press_energy pre-filter half-width (eV); pairs beyond
#                       this are assumed non-duplicate without a geometry check.
# ---------------------------------------------------------------------------

DEDUP_PRESETS: dict[str, dict[str, float]] = {
    "loose":  dict(len_rtol=0.50, cellpar_angle_tol=10.0, pos_tol_frac=0.10, angle_tol_deg=15.0, energy_tol=0.05),
    "normal": dict(len_rtol=0.35, cellpar_angle_tol=8.0,  pos_tol_frac=0.07, angle_tol_deg=10.0, energy_tol=0.02),
    "rigid":  dict(len_rtol=0.15, cellpar_angle_tol=5.0,  pos_tol_frac=0.03, angle_tol_deg=5.0,  energy_tol=0.005),
}


def resolve_dedup_preset(strictness: str) -> dict[str, float]:
    """
    Look up a named strictness preset (case-insensitive).

    Args:
        strictness: One of "loose", "normal", "rigid".

    Returns:
        Preset dict with keys len_rtol, cellpar_angle_tol, pos_tol_frac,
        angle_tol_deg, energy_tol.
    """
    key = strictness.lower()
    if key not in DEDUP_PRESETS:
        raise ValueError(
            f"Unknown dedup strictness '{strictness}', expected one of {list(DEDUP_PRESETS)}"
        )
    return DEDUP_PRESETS[key]


def molecule_fingerprints(
    xtal: Atoms, natoms: int, nmol: int, ref_mol_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a per-structure fingerprint: cell parameters, and each molecule's
    fractional center of geometry + best-fit orientation vs the reference
    molecule.

    Args:
        xtal: full optimized crystal (nmol*natoms atoms).
        natoms: atoms per molecule.
        nmol: molecules per unit cell.
        ref_mol_positions: standardized reference molecule positions (natoms, 3).

    Returns:
        (cellpar, cogs_frac, rots, L): cellpar is (a,b,c,alpha,beta,gamma);
        cogs_frac is (nmol,3); rots is (nmol,3,3) -- each is the best-fit
        ORTHOGONAL matrix (see _orthogonal_fit), not necessarily a proper
        rotation (det can be -1 for a molecule generated via an
        inversion/mirror space-group operation); L is the (3,3) Cartesian
        lattice matrix (rows = a,b,c vectors), needed by is_fast_duplicate
        to test axis-relabeling equivalence.
    """
    cellpar = xtal.cell.cellpar()
    L = xtal.cell.array
    pos = xtal.positions
    cogs_frac = np.zeros((nmol, 3))
    rots = np.zeros((nmol, 3, 3))
    for m in range(nmol):
        mol_pos = pos[m * natoms : (m + 1) * natoms]
        cog = mol_pos.mean(axis=0)
        cogs_frac[m] = np.linalg.solve(L.T, cog)
        centered = mol_pos - cog
        rots[m] = _orthogonal_fit(centered, ref_mol_positions)
    return cellpar, cogs_frac, rots, L


def _orthogonal_fit(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """
    Best-fit ORTHOGONAL matrix R (det = +1 or -1) minimizing
    sum ||target_i - R @ source_i||^2, i.e. R @ source ~= target.

    Unlike scipy's Rotation.align_vectors (which only ever returns a proper
    rotation, det=+1), this allows det=-1. That matters here because a
    molecule generated by an inversion-center or mirror-plane space-group
    operation is related to the reference molecule by an IMPROPER
    transform: forcing a proper-rotation fit onto such a molecule produces
    a high-RMSD, essentially meaningless matrix (observed RMSD ~11.8 A on
    real data for molecules generated via such an operation, vs ~0 for the
    correct improper fit) -- which in turn makes is_fast_duplicate's
    relative-orientation consistency check spuriously fail even when the
    molecule's position and the rest of the structure agree exactly.

    Standard (unconstrained) Kabsch/Procrustes: H = source^T @ target,
    SVD H = U S V^T, R = V @ U^T. (Skip the usual det-correction step that
    Kabsch normally applies to force a proper rotation.)
    """
    h = source.T @ target
    u, _s, vt = np.linalg.svd(h)
    return vt.T @ u.T


_CELLPAR_ANGLE_AXIS_PAIRS = [(1, 2), (0, 2), (0, 1)]  # index0=alpha(b,c), 1=beta(a,c), 2=gamma(a,b)


def _permuted_cellpar_angles(angles: np.ndarray, perm: list[int]) -> np.ndarray:
    """Re-express (alpha,beta,gamma) under a relabeling of which axis is a/b/c."""
    out = np.zeros(3)
    for idx, (k, l) in enumerate(_CELLPAR_ANGLE_AXIS_PAIRS):
        pk, pl = perm[k], perm[l]
        out[idx] = angles[3 - pk - pl]
    return out


def _frac_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-image fractional displacement a-b, wrapped to [-0.5, 0.5)."""
    d = a - b
    d -= np.round(d)
    return d


_AXIS_PERMUTATIONS = list(itertools.permutations(range(3)))


def _match_in_shared_frame(
    cogs1: np.ndarray,
    rots1: np.ndarray,
    cogs2: np.ndarray,
    rots2: np.ndarray,
    pos_tol_frac: float,
    angle_tol_deg: float,
) -> bool:
    """
    Given cogs2/rots2 already expressed in structure 1's lattice-axis
    labeling and Cartesian frame, try aligning fp1's molecule 0 to each
    molecule of fp2 in turn (since either structure's "molecule 0" is an
    arbitrary choice among the nmol symmetry-equivalent molecules), match
    the remaining molecules by nearest fractional distance, then check
    orientation consistency.

    Orientation check: two independent BFGS runs can converge to the
    physically identical crystal packing but with the whole crystal rigidly
    moved by an arbitrary rotation (nothing pins down absolute orientation
    once the lattice-axis labeling and frame are already aligned). The
    correct invariant is that the *relative* transform between every
    matched molecule pair (rots2[k] @ rots1[i].T) agrees with a single
    common rotation G across all nmol pairs -- i.e. the whole assembly
    moved together rigidly. With nmol==1 there is nothing to compare a
    rotation against (a lone asymmetric-unit molecule's absolute
    orientation carries no crystallographic meaning), so position agreement
    alone is sufficient.
    """
    nmol = cogs1.shape[0]
    for j in range(nmol):
        shift = _frac_dist(cogs2[j], cogs1[0])
        shifted1 = cogs1 + shift

        remaining = list(range(nmol))
        matched_pairs = []
        ok = True
        for i in range(nmol):
            best_k, best_d = None, None
            for k in remaining:
                dist = np.linalg.norm(_frac_dist(shifted1[i], cogs2[k]))
                if best_d is None or dist < best_d:
                    best_d, best_k = dist, k
            if best_d is None or best_d > pos_tol_frac:
                ok = False
                break
            remaining.remove(best_k)
            matched_pairs.append((i, best_k))
        if not ok:
            continue

        if nmol == 1:
            return True

        # rots1/rots2 entries can be improper (det=-1) for molecules
        # generated via an inversion/mirror space-group operation (see
        # _orthogonal_fit). The pairwise interaction potential used by
        # RigidPressSymm only depends on interatomic distances, so it is
        # blind to chirality: a structure and its whole-crystal mirror
        # image are physically the same packing (same energy, same
        # cellpar) even though every molecule's fitted orthogonal matrix
        # flips handedness relative to the un-mirrored structure. A valid
        # correspondence therefore requires det(rel) to have the SAME sign
        # across all matched pairs (all +1: shared global rotation; all -1:
        # shared global rotation+reflection) -- not a sign mismatch, and
        # not necessarily all +1. When consistent, rel @ ref_rel.T always
        # has det=+1 (product of two equal-sign values), safe to feed to
        # Rotation.from_matrix below.
        rels = [rots2[k] @ rots1[i].T for i, k in matched_pairs]
        dets = [np.linalg.det(rel) for rel in rels]
        if not (all(dd > 0 for dd in dets) or all(dd < 0 for dd in dets)):
            continue
        ref_rel = rels[0]
        consistent = True
        for rel in rels[1:]:
            diff = rel @ ref_rel.T
            ang = Rotation.from_matrix(diff).magnitude() * 180 / np.pi
            if min(ang, 360 - ang) > angle_tol_deg:
                consistent = False
                break
        if consistent:
            return True

    return False


_PERM_ARRAY = np.array(_AXIS_PERMUTATIONS)  # (6,3)
_PERM_ANGLE_IDX = np.array(
    [[3 - perm[k] - perm[l] for (k, l) in _CELLPAR_ANGLE_AXIS_PAIRS] for perm in _AXIS_PERMUTATIONS]
)  # (6,3), precomputed once: which of (alpha,beta,gamma) each permuted angle-slot maps to


def is_fast_duplicate(
    fp1: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fp2: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pos_tol_frac: float = DEDUP_PRESETS["normal"]["pos_tol_frac"],
    angle_tol_deg: float = DEDUP_PRESETS["normal"]["angle_tol_deg"],
    cellpar_len_rtol: float = DEDUP_PRESETS["normal"]["len_rtol"],
    cellpar_angle_tol: float = DEDUP_PRESETS["normal"]["cellpar_angle_tol"],
    lattice_fit_rtol: float = 0.03,
) -> bool:
    """
    Compare two molecule_fingerprints() results for the same spg group.

    High-symmetry lattices (orthorhombic and above) can be described with
    any of 6 axis relabelings (which lattice vector is called a/b/c) that
    all give the identical physical crystal -- two independent BFGS runs
    have no reason to converge to the same labeling. This searches all 6
    permutations of structure 2's lattice vectors; for each, it uses a
    Kabsch fit on the 3 lattice vectors to find the best orthogonal
    transform R mapping structure 2's Cartesian frame onto structure 1's
    (this also subsumes the identity-permutation case of a pure global
    rotation/reflection between two otherwise-identically-labeled cells).
    A residual fit check confirms R genuinely aligns the two lattices (not
    just a coincidental length/angle match), then structure 2's molecule
    fingerprints are re-expressed in structure 1's frame via R + the axis
    permutation before delegating to _match_in_shared_frame.

    Performance note (profiled: this was ~50% of total dedup time):
    the naive version calls np.linalg.svd once per permutation in a Python
    loop, but numpy's generic SVD has fixed per-call dispatch overhead that
    dwarfs the actual FLOPs for a 3x3 matrix. Both the length/angle
    pre-filter and the Kabsch fit are vectorized here across all 6
    permutations at once (one batched np.linalg.svd call on a (k,3,3)
    stack instead of up to 6 separate (3,3) calls), which is the same math
    as looping but avoids paying per-call overhead 6 times.
    """
    cellpar1, cogs1, rots1, L1 = fp1
    cellpar2, cogs2, rots2, L2 = fp2
    lengths1 = cellpar1[:3]
    scale = np.mean(lengths1)

    L2_perms = L2[_PERM_ARRAY]  # (6,3,3): L2_perms[p] == L2[_AXIS_PERMUTATIONS[p], :]
    lengths2_perms = np.linalg.norm(L2_perms, axis=2)  # (6,3)
    length_ok = np.all(
        np.abs(lengths1[None, :] - lengths2_perms) / ((lengths1[None, :] + lengths2_perms) / 2) <= cellpar_len_rtol,
        axis=1,
    )

    angles2_perms = cellpar2[3:6][_PERM_ANGLE_IDX]  # (6,3)
    angle_ok = np.all(np.abs(cellpar1[3:6][None, :] - angles2_perms) <= cellpar_angle_tol, axis=1)

    candidates = np.nonzero(length_ok & angle_ok)[0]
    if candidates.size == 0:
        return False

    L2c = L2_perms[candidates]  # (k,3,3)
    H = np.matmul(np.transpose(L2c, (0, 2, 1)), L1[None, :, :])  # H[p] = L2c[p].T @ L1
    u, _s, vt = np.linalg.svd(H)  # batched over leading dim k
    R = np.matmul(np.transpose(vt, (0, 2, 1)), np.transpose(u, (0, 2, 1)))  # R[p] = vt[p].T @ u[p].T

    recon = np.matmul(L2c, np.transpose(R, (0, 2, 1)))
    resid = np.sqrt(np.mean(np.sum((L1[None, :, :] - recon) ** 2, axis=2), axis=1))
    valid = np.nonzero(resid / scale <= lattice_fit_rtol)[0]

    for vi in valid:
        perm = _AXIS_PERMUTATIONS[candidates[vi]]
        Rp = R[vi]
        cogs2_frame = cogs2[:, list(perm)]
        rots2_frame = np.einsum("ij,mjk->mik", Rp, rots2)
        if _match_in_shared_frame(cogs1, rots1, cogs2_frame, rots2_frame, pos_tol_frac, angle_tol_deg):
            return True

    return False


def fast_dedup_bucket(
    bucket: dict[str, Atoms],
    ref_mol_positions: np.ndarray,
    natoms: int,
    nmol: int,
    energy_key: str | None,
    energy_tol: float = DEDUP_PRESETS["normal"]["energy_tol"],
    pos_tol_frac: float = DEDUP_PRESETS["normal"]["pos_tol_frac"],
    angle_tol_deg: float = DEDUP_PRESETS["normal"]["angle_tol_deg"],
    cellpar_len_rtol: float = DEDUP_PRESETS["normal"]["len_rtol"],
    cellpar_angle_tol: float = DEDUP_PRESETS["normal"]["cellpar_angle_tol"],
) -> dict[str, Atoms]:
    """
    Fast fingerprint-based equivalent of dedup_bucket: deduplicate a single
    volume bucket using molecule_fingerprints()/is_fast_duplicate() instead
    of pymatgen's StructureMatcher.

    Args:
        bucket: {name: Atoms} structures in this volume bucket (same spg).
        ref_mol_positions: standardized reference molecule positions (natoms, 3).
        natoms: atoms per molecule.
        nmol: molecules per unit cell.
        energy_key: key in Atoms.info for energy, used both to skip obviously
            non-duplicate pairs and to pick the best of a duplicate cluster.
        energy_tol, pos_tol_frac, angle_tol_deg, cellpar_len_rtol,
            cellpar_angle_tol: strictness thresholds, see DEDUP_PRESETS.

    Returns:
        {name: Atoms} unique structures.
    """
    fps = {name: molecule_fingerprints(x, natoms, nmol, ref_mol_positions) for name, x in bucket.items()}

    pool = set(bucket.keys())
    kept = {}
    while pool:
        ref_name = next(iter(pool))
        pool.discard(ref_name)
        e_ref = bucket[ref_name].info.get(energy_key) if energy_key else None
        if e_ref is not None:
            e_ref = float(e_ref)
            if math.isinf(e_ref):
                e_ref = None

        cluster = {ref_name: bucket[ref_name]}
        for name in list(pool):
            if e_ref is not None:
                e = bucket[name].info.get(energy_key)
                if e is not None:
                    e = float(e)
                    if not math.isinf(e) and abs(e - e_ref) > energy_tol:
                        continue
            if is_fast_duplicate(
                fps[ref_name], fps[name],
                pos_tol_frac=pos_tol_frac, angle_tol_deg=angle_tol_deg,
                cellpar_len_rtol=cellpar_len_rtol, cellpar_angle_tol=cellpar_angle_tol,
            ):
                cluster[name] = bucket[name]
                pool.discard(name)

        best = _select(cluster, energy_key)
        kept[best] = cluster[best]

    return kept
