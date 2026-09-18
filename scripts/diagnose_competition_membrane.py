"""Build two explicit POPE systems for a bounded technical MD pilot, not activity ranking."""

import argparse
import hashlib
import importlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def surface_coordinates(coordinates: np.ndarray) -> np.ndarray:
    result = np.asarray(coordinates, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("Finite Cartesian coordinates required")
    result[:, :2] -= result[:, :2].mean(0)
    result[:, 2] += 2.0 - result[:, 2].min()
    return result


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def lipid_heavy_atoms(atoms: list[Any]) -> list[int]:
    # The bundled POPE template uses the three-character PDB residue name POP.
    indices = [a.index for a in atoms if a.residue.name == "POP" and a.element.symbol != "H"]
    if not indices:
        raise ValueError("No lipid heavy atoms in the POPE system")
    return indices


def contacts(
    x: np.ndarray, peptide: list[int], lipid: list[int], box: np.ndarray
) -> dict[str, Any]:
    minima = []
    for index in peptide:
        delta = x[lipid] - x[index]
        delta -= box * np.rint(delta / box)
        minima.append(float(np.linalg.norm(delta, axis=1).min()))
    return dict(
        minimum_lipid_distance_nm=min(minima),
        peptide_atoms_within_0_45nm=sum(d < 0.45 for d in minima),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path("work/competition_exploration/20260912-b")
    structure = root / "phase4/structure-r2"
    folds = json.loads((structure / "manifest.json").read_text())
    chosen = []
    for cohort in ["alternative", "control"]:
        eligible = [
            r for r in folds["results"] if r["cohort"] == cohort and "C" not in r["sequence"]
        ]
        if not eligible:
            raise ValueError("No cysteine-free folded candidate in cohort")
        chosen.append(eligible[0])
    openmm = importlib.import_module("openmm")
    app = importlib.import_module("openmm.app")
    unit = importlib.import_module("openmm.unit")
    fixer_class = importlib.import_module("pdbfixer").PDBFixer
    if app.__file__ is None:
        raise ValueError("OpenMM data location is unavailable")
    data = Path(app.__file__).parent / "data"
    inputs = [
        Path(__file__),
        structure / "manifest.json",
        data / "amber14-all.xml",
        data / "POPE.pdb",
        *sorted((data / "amber14").glob("*.xml")),
        root / "envs/membrane/uv.lock",
    ]
    maximum_wall_seconds = 7200 - float(folds["seconds"])
    protocol = dict(
        seed=42,
        candidates=chosen,
        forcefield=["amber14-all.xml", "amber14/tip3p.xml"],
        membrane="pure POPE, generic inner-membrane toy system; not species-specific/LPS",
        water="TIP3P",
        salt_molar=0.15,
        ph=7.4,
        temperature_kelvin=300,
        timestep_fs=1,
        steps=2000,
        duration_ps=2,
        ensemble="NVT Langevin-middle; no production equilibration",
        placement="translate intact fold: XY center0, lowest atom z=2nm above bilayer center0",
        scope="construction and finite-trajectory smoke; no binding, pore or MIC inference",
        maximum_wall_seconds=maximum_wall_seconds,
        input_sha256={str(p): sha(p) for p in inputs},
        platform="CUDA mixed",
    )
    write(args.output / "protocol.json", protocol)
    platform = openmm.Platform.getPlatformByName("CUDA")
    platform.setPropertyDefaultValue("Precision", "mixed")
    platform.setPropertyDefaultValue("DeterministicForces", "true")
    started = time.monotonic()
    records = []
    for candidate in chosen:
        if time.monotonic() - started > maximum_wall_seconds:
            raise TimeoutError("Combined structure/membrane pilot budget exceeded")
        random.seed(42)
        np.random.seed(42)
        path = args.output / candidate["cohort"]
        path.mkdir()
        input_pdb = structure / f"{candidate['index']:02}.pdb"
        fixer = fixer_class(filename=str(input_pdb))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=42)
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        model = app.Modeller(fixer.topology, fixer.positions)
        model.addHydrogens(forcefield, pH=7.4, platform=platform)
        peptide_count = model.topology.getNumAtoms()
        model.positions = (
            surface_coordinates(model.positions.value_in_unit(unit.nanometer)) * unit.nanometer
        )
        model.addMembrane(
            forcefield,
            lipidType="POPE",
            minimumPadding=1 * unit.nanometer,
            ionicStrength=0.15 * unit.molar,
            platform=platform,
        )
        system = forcefield.createSystem(
            model.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1 * unit.nanometer,
            constraints=app.HBonds,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.001 * unit.picoseconds
        )
        integrator.setRandomNumberSeed(42)
        simulation = app.Simulation(
            model.topology,
            system,
            integrator,
            platform,
            {"Precision": "mixed", "DeterministicForces": "true"},
        )
        simulation.context.setPositions(model.positions)
        simulation.minimizeEnergy(maxIterations=500)
        initial = simulation.context.getState(getPositions=True, getEnergy=True)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 42)
        simulation.reporters.append(app.DCDReporter(str(path / "trajectory.dcd"), 100))
        simulation.reporters.append(
            app.StateDataReporter(
                str(path / "states.csv"),
                100,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
            )
        )
        simulation.step(2000)
        final = simulation.context.getState(getPositions=True, getEnergy=True)
        x0 = initial.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        x1 = final.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        energy = final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(x1).all() or not np.isfinite(energy):
            raise ValueError("Nonfinite membrane trajectory")
        atoms = list(model.topology.atoms())
        peptide = [a.index for a in atoms[:peptide_count] if a.element.symbol != "H"]
        lipid = lipid_heavy_atoms(atoms)
        box = np.diag(final.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
        with (path / "final.pdb").open("w") as stream:
            app.PDBFile.writeFile(model.topology, final.getPositions(), stream)
        (path / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
        (path / "integrator.xml").write_text(openmm.XmlSerializer.serialize(integrator))
        simulation.saveState(str(path / "final_state.xml"))
        record = dict(
            cohort=candidate["cohort"],
            sequence=candidate["sequence"],
            atoms=len(atoms),
            residues=dict(Counter(r.name for r in model.topology.residues())),
            potential_energy_kj_mol=float(energy),
            initial=contacts(x0, peptide, lipid, box),
            final=contacts(x1, peptide, lipid, box),
            input_pdb_sha256=sha(input_pdb),
        )
        records.append(record)
        write(path / "result.json", record)
        print(record, flush=True)
        del simulation, integrator, system
    write(
        args.output / "manifest.json",
        dict(
            **protocol,
            results=records,
            seconds=time.monotonic() - started,
            artifacts_sha256={
                str(p.relative_to(args.output)): sha(p)
                for p in args.output.rglob("*")
                if p.is_file()
            },
        ),
    )


if __name__ == "__main__":
    main()
