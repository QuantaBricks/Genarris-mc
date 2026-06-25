"""
GFN-FF energy stub used by XTBOptimizer for xtb-based cell optimization.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from ase import Atoms
from gnrs.core.energy import EnergyCalculatorABC


class GFNFFEnergy(EnergyCalculatorABC):
    """
    Holds xtb binary settings for GFN-FF optimization via XTBOptimizer.
    XTBOptimizer calls xtb directly as a subprocess; this class is not
    used to evaluate forces or energies itself.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.xtb_bin = self.tsk_set.get("command", "xtb")
        self.xtb_path = self.tsk_set.get("xtb_path", "")
        # Return self so XTBOptimizer can reach xtb_bin / xtb_path.
        self.calc = self

    def initialize(self) -> None:
        pass

    def compute(self, xtal: Atoms) -> None:
        pass

    def finalize(self) -> None:
        pass
