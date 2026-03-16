# ARID-sf — Scoring Tool

## Usage

```bash
python score_round.py path/to/modelsFolders my_output.csv
```

Where `path/to/modelsFolders` is a directory containing one or more model subdirectories, structured as follows:

```
path/to/modelsFolders/
├── 1a14/
│   ├── 1a14_001.pdb
│   ├── 1a14_002.pdb
│   └── ...
└── 5yy1/
    ├── 5yy1_001.pdb
    └── 5yy1_002.pdb
```

> **Important:** Each subfolder (e.g. `1a14`, `5yy1`) must contain models from **one system only** (the same antibody and antigen). If you have multiple different systems (e.g. when testing ARID on solved structures), use `score_refs.py` instead.

> **Note:** ARID processes each subfolder independently and parallelizes tasks across all models within it. Performance is optimal when each subfolder contains thousands of models and multiple CPUs are available.

---

## Parameters

To customize the computation, edit the following variables in `score_round.py`:

| Parameter | Default | Description |
|---|---|---|
| `n_workers` | `40` | Number of CPU workers |
| `batch_size` | `1000` | Number of models per batch |
| `feature_cap` | `1000` | Feature boundaries in case of clashes — **do not change** |
| `cap_length` | `75` | Max number of residues considered per model interface — **do not change** |
| `CAP_MEMORY` | `1000` | Number of models processed before writing to disk (RAM-dependent) |

---

## Output

A `.csv` file with the following structure:

```
Model,pDockQ,pFNAT_env,pJACA,pJACB
1A14_1_4908_R_4908,0.29489622,0.503456,0.65096414,0.61525863
1A14_1_5389_R_5389_ti,0.11942382,0.34318617,0.44313583,0.4278602
...
```

- **Model**: the scored file name
- **pDockQ**: the predicted ARID-sf score
  - A score close to **1** indicates a good prediction
  - A score close to **0** indicates an incorrect prediction

---

## Input File Format

For the program to work correctly, each `.pdb` file must respect the following requirements:

- The file must be in `.pdb` format
- All **antigen** chains must be renamed `"A"` (even if there are multiple chains)
- All **antibody** chains must be renamed `"B"` (even if there are multiple chains)
- Antigen atoms/chains (`"A"`) must appear **first** in the file
- Antibody atoms/chains (`"B"`) must appear **second** (after the antigen) in the file
- Residues must be renumbered from `1` to `N`, where `N = antigen residues + antibody residues`
- Atoms must follow the **OPLS United Atom (UA) forcefield** nomenclature (see below)

### Using ARID with HADDOCK3 Poses

Make sure to run HADDOCK with the antigen chain as `"A"` and the antibody chain as `"B"`.

If this is the case, you can run ARID-sf directly with HADDOCK3 models.

HADDOCK3 file preparation relies on tools from [pdb-tools](https://www.bonvinlab.org/education/HADDOCK3/HADDOCK3-antibody-antigen/).

### OPLS UA Nomenclature

Only polar hydrogens are present. Atom names and residue type names can be found in `/src/ARIDv2.0/lookup_dict.py`.

An easy way to convert an antibody-antigen complex `.pdb` file to OPLS UA format is to use the HADDOCK3 `[topoaa]` module. An example script is available at `src/benchmark_sets/model_to_ua.py`.

### Helper Formatting Scripts

A set of helper scripts is available in `./formatting`. These allow you to:
- Rechain and renumber PDB files
- Run the HADDOCK3 topology module to obtain OPLS UA nomenclature
- Parse HADDOCK3 output and copy ARID-sf-ready files into a target directory

For creaating correct topologies, these scripts require [HADDOCK3](https://github.com/haddocking/haddock3.git).

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/DSIMB/ARID-sf.git
cd ARID-sf
```

**2. Create and activate a conda environment:**
```bash
mamba create -n arid-env python=3.11 -c conda-forge
mamba activate arid-env
```

**3. Install dependencies:**
```bash
mamba install -c conda-forge numpy=1.23.5 pandas=2.3.1 httpx cython matplotlib pyparsing biotite mdtraj
```

**4. Install PyTorch and CUDA (pytorch 2.8.0+cu128):**

See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the right command for your machine. For example:
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

**5. Install ESM-C:**
```bash
pip install esm==3.2.1
```
> **Note:** pip may report an error, but the required packages should install successfully.

**6. Build the Cython extension:**
```bash
cd ARIDv2.0
python setup_v2.py build_ext --inplace
cd ..
```

**7. Test your installation:**
```bash
python3 ARIDv2.0/score_round.py example/models example/outputs/example_output.csv
```