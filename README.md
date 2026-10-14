# Open-pit mine scheduling: At and By QUBO formulations with BF-DCQO

Research code and figure-generation scripts supporting a comparative study of two QUBO encodings for open-pit mine production scheduling, evaluated with bias-field digitized counterdiabatic quantum optimization (BF-DCQO) using Qiskit Aer simulation.

**Start here:** [Read the mathematical formulation and implementation details](Manuscript/Formulation.md).

## Repository structure

```text
Manuscript/
    Formulation.md               Mathematical model and correspondence to code

experiments/
    fixed/                       Fixed 22-qubit case, At and By
    random/                      100 random 22-qubit instances, At and By, s=0

figures/
    diagrams/                    Conceptual manuscript figures 01, 02, 03, 05
    results/                     Experimental figures 06–09

data/                            Experimental JSON results
outputs/                         Generated figures (PDF and PNG)

requirements.txt                 Python dependencies and pinned versions
README.md                        Repository documentation
```

## Requirements

The numerical experiments were run in a **Linux Conda environment**. The environment package listing reports **Python 3.12.12**, with a separate `cpython` package at **3.12.13**; confirm the active interpreter using `python --version` before preparing the archival release. The principal recorded package versions are:

| Package | Version |
|---|---|
| NumPy | 2.5.0 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.0 |
| Qiskit | 2.4.2 |
| Qiskit Aer | 0.17.2 |
| GurobiPy | 13.0.3 |

The pinned Python package requirements are listed in [`requirements.txt`](requirements.txt). They are based on the Linux Conda package inventory, **not yet validated as a fresh pip installation**. The Conda inventory reports a `py310` build tag for `qiskit` alongside Python 3.12 packages, so confirm the imported Qiskit version in the environment before claiming exact reproducibility.

Random-instance scripts additionally require a working Gurobi license.

The experiments use Qiskit Aer for classical simulation of quantum circuits. CPU execution is supported; GPU execution requires a compatible CUDA-enabled Qiskit Aer installation.

### Environment setup

For a new environment, use a compatible Python 3.12 installation. For the closest match to the research environment, prefer Conda on Linux; the commands below describe a pip-based setup, not an exact recreation of the Conda build configuration.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Alternatively, create or use a Conda environment on Linux with the documented versions. Because the recorded Gurobi package is `gurobi` from the Gurobi Conda channel, installing `gurobipy` through pip may differ from the original setup. Check licensing separately. Before releasing an exact environment specification, export and validate a Conda `environment.yml` from the experiment machine.

## Running the experiments

The repository contains separate experiments for a fixed mining instance and a collection of randomly generated instances.

### Fixed-case experiments

The fixed case contains 22 QUBO variables and compares the At and By formulations.

```bash
cd experiments/fixed
python At_22Q_FixedCase.py
python By_22Q_FixedCase.py
```

The resulting JSON files are stored in `data/`:

- `At_results_independent_R.json`
- `By_results_independent_R.json`

### Random-instance experiments

The random experiments evaluate 100 independently generated 22-qubit instances for both formulations, without greedy postprocessing (`s=0`).

```bash
cd experiments/random
python At_22Q_RandomInstances_s0.py
python By_22Q_RandomInstances_s0.py
```

The resulting JSON files are stored in `data/`:

- `At_22Q_random_100_independent_R_s0.json`
- `By_22Q_random_100_independent_R_s0.json`

**Note:** These experiments can be computationally expensive. Review the execution parameters in the respective `main()` functions before running them.

## Reproducing the figures

All figure-generation scripts are located in `figures/`, and generated files are saved under `outputs/`.

Figure output paths are resolved relative to each script's location rather than the current working directory.

### Conceptual figures

Located in `figures/diagrams/`:

| Script | Description |
|---|---|
| `Figure01.py` | Open-pit mine scheduling concepts |
| `Figure02.py` | Qubit indexing and variable ordering |
| `Figure03.py` | Interaction graphs of the At and By formulations |
| `Figure05.py` | Illustrative optimal mining schedule |

These scripts generate conceptual figures independently of the experimental JSON results.

### Experimental figures

Located in `figures/results/`:

| Script | Figure | Description |
|---|---|---|
| `Figure06_07.py` | Figure 6 | Optimal-sampling probability in the fixed case |
| `Figure06_07.py` | Figure 7 | Terminal QUBO energy histogram in the fixed case |
| `Figure08_09.py` | Figure 8 | Median and interquartile range across random instances |
| `Figure08_09.py` | Figure 9 | Paired At versus By comparison across random instances |

After generating the corresponding JSON files:

```bash
cd figures/results
python Figure06_07.py
python Figure08_09.py
```

The resulting plots are saved in `outputs/`.

## Scientific interpretation

The fixed-case experiments and random-instance experiments use different postprocessing settings.

The fixed-case experiments include greedy postprocessing analyses with sweep counts ranging from `s=0` to `s=10`.

The random-instance experiments use **no greedy refinement (`s=0`)** and report raw terminal optimal-sampling probabilities.

These quantities must not be conflated when comparing experimental results.

All quantum circuits in this repository are evaluated using **classical quantum-circuit simulation**, not quantum hardware.

## Reproducibility

To reproduce the numerical experiments, researchers should use the documented software versions, experimental parameters, and corresponding JSON datasets.

For an archival publication release, we recommend preserving:

- The exact source-code revision.
- Python and package versions.
- Random seeds and experimental parameters.
- The numerical JSON results used to generate published figures.
- The corresponding figure-generation scripts.

## Citation and archival release

If you use this repository, please cite the associated journal article once its bibliographic details are available.

A `CITATION.cff` file, article DOI, author information, and repository license should be added after confirmation with all contributors.
