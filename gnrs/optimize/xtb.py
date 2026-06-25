"""
Geometry optimizer that drives xtb GFN-FF cell optimization as a subprocess.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import os
import re
import subprocess
import logging

from ase import Atoms
from ase.io import write, read

from gnrs.core.optimizer import GeometryOptimizerABC

logger = logging.getLogger("xtb_optimizer")

_EH_TO_EV = 27.211386245988


class XTBOptimizer(GeometryOptimizerABC):
    """
    Runs xtb --gfnff --opt via subprocess; xtb's internal ancopt drives the
    optimization.  Each MPI rank writes to its own rank/struct subdirectory.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.opt_name = "xtb"
        self.maxcycle = int(self.tsk_set.pop("maxcycle", 200))
        # energy_calc is a GFNFFEnergy instance (returns self from get_calculator)
        self.xtb_bin = self.energy_calc.xtb_bin
        self.xtb_path = self.energy_calc.xtb_path
        self._struct_counter = 0

    def optimize(self, xtal: Atoms) -> None:
        self._struct_counter += 1
        work_dir = os.path.join(f"rank_{self.rank}", f"struct_{self._struct_counter}")
        os.makedirs(work_dir, exist_ok=True)

        poscar_path = os.path.join(work_dir, "POSCAR")
        write(poscar_path, xtal, format="vasp")

        xcontrol_path = os.path.join(work_dir, "xcontrol")
        with open(xcontrol_path, "w") as f:
            f.write(f"$opt\n   maxcycle={self.maxcycle}\n$end\n")

        env = os.environ.copy()
        if self.xtb_path:
            env["XTBPATH"] = self.xtb_path
        env["OMP_NUM_THREADS"] = "1,1"
        env["MKL_NUM_THREADS"] = "1"

        cmd = [
            self.xtb_bin, "POSCAR",
            "--gfnff", "--opt",
            "-I", "xcontrol",
            "-P", "1",
            "--norestart",
        ]

        result = subprocess.run(
            cmd, cwd=work_dir, env=env,
            capture_output=True, text=True,
        )

        self._energy = 0.0
        self.converged = False

        for line in result.stdout.splitlines():
            if "TOTAL ENERGY" in line:
                m = re.search(r"TOTAL ENERGY\s+([-\d.]+)\s+Eh", line)
                if m:
                    self._energy = float(m.group(1)) * _EH_TO_EV

        for line in result.stderr.splitlines():
            if "normal termination" in line:
                self.converged = True

        if result.returncode != 0 and not self.converged:
            logger.warning(
                "xtb rank %d struct %d returned code %d",
                self.rank, self._struct_counter, result.returncode,
            )

        opt_poscar = os.path.join(work_dir, "xtbopt.poscar")
        if os.path.exists(opt_poscar):
            opt = read(opt_poscar, format="vasp")
            xtal.positions = opt.positions
            xtal.cell = opt.cell
        else:
            logger.warning(
                "xtbopt.poscar not found for rank %d struct %d",
                self.rank, self._struct_counter,
            )

    def update(self, xtal: Atoms) -> None:
        xtal.info[f"{self.opt_name}_{self.energy_method}"] = self._energy
        xtal.info[self.opt_name] = "converged" if self.converged else "unconverged"

    def finalize(self, xtal: Atoms) -> None:
        xtal.calc = None
