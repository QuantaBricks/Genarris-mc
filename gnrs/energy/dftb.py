"""
This module computes the energy using DFTB+ code with ASE.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os

from ase import Atoms
from ase.calculators.dftb import Dftb

from gnrs.core.energy import EnergyCalculatorABC


class _DftbV12(Dftb):
    """Dftb calculator that writes ParserVersion = 12 instead of 1."""

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)
        hsd_path = os.path.join(self.directory, "dftb_in.hsd")
        with open(hsd_path) as f:
            content = f.read()
        content = content.replace("ParserVersion = 1", "ParserVersion = 12")
        with open(hsd_path, "w") as f:
            f.write(content)


class DFTBEnergy(EnergyCalculatorABC):
    """
    Computes energy using DFTB+ code.
    https://wiki.fysik.dtu.dk/ase/ase/calculators/dftb.html
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        sk_files = self.tsk_set["sk_files"].rstrip("/") + "/"
        self.tsk_set["energy_settings"]["slako_dir"] = sk_files
        self.tsk_set["energy_settings"]["command"] = self.tsk_set["command"]
        # each rank writes to its own subdirectory to avoid file conflicts
        rank_dir = f"rank_{self.rank}"
        self.tsk_set["energy_settings"]["directory"] = rank_dir
        self.calc = _DftbV12(**self.tsk_set["energy_settings"])

    def initialize(self) -> None:
        pass

    def compute(self, xtal: Atoms) -> None:
        xtal.calc = self.calc
        try:
            energy = xtal.get_potential_energy()
        except Exception:
            energy = 0
        xtal.info[self.energy_name] = energy

    def finalize(self) -> None:
        pass
