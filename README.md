# XAI Mini Project: Global DL Explanations for RDF GNNs

This project implements Strategy 3 from the XAI mini project: train an R-GCN on an RDF dataset, convert the GNN predictions into positive and negative examples, and learn global description-logic explanations with CELOE from Ontolearn.

The code is configuration-driven. The same pipeline runs three experiments:

| Experiment | Config | Purpose |
| --- | --- | --- |
| AIFB | `configs/aifb.yaml` | Research group prediction on AIFB |
| MUTAG | `configs/mutag.yaml` | Mutagenicity prediction using RDF type/literal node features |
| MUTAG node-id ablation | `configs/mutag_node_id.yaml` | Ablation without RDF-derived initial features |

## Repository Layout

```text
configs/                  Experiment configs
data/                     RDF datasets and train/test label splits
docs/                     Course/project reference material
src/xai_miniproject/      Python package
tests/                    Small regression tests
artifacts/                Generated outputs, ignored by git
logs/                     Timestamped terminal logs, ignored by git
```

Important generated files are written under `artifacts/<experiment>/`:

```text
dataset_stats.json
metrics.json
predictions.csv
model.pt
learning_problems.json
explanation_results.json
explanation_results.csv
```

Each CLI run also writes a terminal log:

```text
logs/aifb_log/YYYYMMDD_HHMMSS_run-all.log
logs/mutag_log/YYYYMMDD_HHMMSS_run-all.log
logs/mutag_node_id_log/YYYYMMDD_HHMMSS_run-all.log
```

## Setup

Use Python 3.10 or newer. Python 3.12 has been tested on macOS.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Conda

```bash
conda create -n xai python=3.12
conda activate xai
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Install Ontolearn / CELOE

Install the CELOE dependencies after the core pipeline works:

```bash
python -m pip install -r requirements-ontolearn.txt
python -m pip install --no-deps ontolearn==0.10.0
```

The two-step install avoids Ontolearn's pinned `python-sat==0.1.7.dev23`, which can fail to compile on macOS arm64 with Python 3.12. The project has been verified with the newer prebuilt `python-sat` wheel installed from `requirements-ontolearn.txt`.

If Ontolearn is not installed, the code falls back to a simple baseline explainer so the pipeline can still run, but the main results should use `backend: ontolearn`.

## Run Experiments

Run the default AIFB experiment:

```bash
xai-mini run-all
```

Run a specific experiment:

```bash
xai-mini --config configs/aifb.yaml run-all
xai-mini --config configs/mutag.yaml run-all
xai-mini --config configs/mutag_node_id.yaml run-all
```

Run individual stages:

```bash
xai-mini --config configs/mutag.yaml analyze
xai-mini --config configs/mutag.yaml train
xai-mini --config configs/mutag.yaml explain
```

Disable terminal log creation for temporary debugging:

```bash
xai-mini --config configs/mutag.yaml --no-log train
```

If the package is not installed with `pip install -e .`, run via module path:

```bash
PYTHONPATH=src python -m xai_miniproject.cli --config configs/mutag.yaml run-all
```

Windows PowerShell equivalent:

```powershell
$env:PYTHONPATH="src"
python -m xai_miniproject.cli --config configs/mutag.yaml run-all
```

## Current Expected Results

The exact numbers can vary slightly across platforms and dependency versions. On the tested macOS conda environment:

| Experiment | Initial features | Test macro-F1 | Explanation backend |
| --- | --- | ---: | --- |
| AIFB | node-id embedding | about 0.84 | Ontolearn CELOE |
| MUTAG | RDF type/literal features | about 0.71 | Ontolearn CELOE |
| MUTAG node-id ablation | node-id embedding | about 0.54 | Ontolearn CELOE |

The MUTAG ablation is intentionally worse: it tests the effect of removing RDF-derived initial node features.

## Development Checks

Install the optional development tools first:

```bash
python -m pip install -e ".[dev]"
```

```bash
python -m compileall -q src
python -m pytest
```
