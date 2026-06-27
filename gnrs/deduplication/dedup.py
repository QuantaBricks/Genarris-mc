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

import logging
import random
from collections import defaultdict

from mpi4py import MPI
from ase.atoms import Atoms
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
    Sub-group structures by unit cell volume into exactly n_buckets equal-width
    bins. Adjacent buckets share a small overlap so duplicates straddling a
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
    edges = [v_min + (v_max - v_min) * i / n for i in range(n + 1)]
    # Expand edges slightly so boundary structures fall inside a bucket.
    edges[0] -= 1e-6
    edges[-1] += 1e-6

    buckets: list[dict[str, Atoms]] = [dict() for _ in range(n)]
    for name, xtal in items:
        vol = xtal.get_volume()
        for k in range(n):
            if edges[k] < vol <= edges[k + 1]:
                buckets[k][name] = xtal
                break

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
) -> dict[str, Atoms]:
    """
    Dispatch one spg pool at a time to worker ranks (rank 0 = dispatcher).
    Each worker receives one spg pool, splits it into volume buckets,
    and deduplicates each bucket sequentially — no MPI inside.

    Args:
        spg_groups: {spg: {name: Atoms}} (only meaningful on rank 0).
        matcher: StructureMatcher instance.
        energy_key: Energy key for selecting best duplicate.

    Returns:
        Combined unique structures (broadcast to all ranks).
    """
    if gp.is_master:
        queue = list(spg_groups.items())  # [(spg, pool), ...]
        kept = {}
        active = gp.size - 1

        total = len(queue)
        done = 0

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
                kept.update(dedup_bucket(bucket, matcher, energy_key))
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
