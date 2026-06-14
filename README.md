# XAI Mini Project: Strategy 3 on RDF Data

This repository implements a runnable mini project pipeline for global explanations of a graph neural network on an RDF dataset. The default configuration uses AIFB and trains a lightweight R-GCN implemented directly in PyTorch. The model predictions are then converted into positive and negative examples for a description logic concept learner such as CELOE from Ontolearn.

The code is configuration driven. Dataset paths, label columns, target classes, and leakage predicates live in YAML files, so switching from AIFB to MUTAG should not require changing the Python source.

## Project Structure

```text
configs/
  aifb.yaml                 # runnable default experiment
  mutag.template.yaml       # template for future MUTAG switch
src/xai_miniproject/
  analyze.py                # RDF and label statistics
  data.py                   # RDF parsing, graph construction, feature extraction
  model.py                  # pure PyTorch R-GCN
  train.py                  # training, evaluation, predictions
  explain.py                # CELOE wrapper plus baseline fallback
  cli.py                    # command line entry point
tests/
  test_config.py
  test_metrics.py
requirements.txt           # core runnable dependencies
requirements-ontolearn.txt # optional CELOE/Ontolearn dependency
```

## Environment Setup

Recommended Python version: 3.10 to 3.12.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Using conda:

```bash
conda create -n xai-mini python=3.12
conda activate xai-mini
python -m pip install -r requirements.txt
python -m pip install -e .
```

For the real CELOE backend, install Ontolearn after the core pipeline works:

```bash
python -m pip install -r requirements-ontolearn.txt
python -m pip install --no-deps ontolearn==0.10.0
```

This two-step install avoids Ontolearn's pinned `python-sat==0.1.7.dev23`, which can fail to compile on macOS arm64 with Python 3.12. We install a newer prebuilt `python-sat` wheel instead. `pip check` may report that Ontolearn asked for the older exact version; the project has been verified with `python-sat 1.9.dev5`.

If you still want to build the old pinned package, install the Xcode command line tools first on macOS. On Windows, WSL or conda is usually the least painful path.

## Reproduce the AIFB Run

The AIFB dataset is already present in `aifb-hetero/`.

Run the complete pipeline:

```bash
xai-mini --config configs/aifb.yaml run-all
```

Or run each stage explicitly:

```bash
xai-mini --config configs/aifb.yaml analyze
xai-mini --config configs/aifb.yaml train
xai-mini --config configs/aifb.yaml explain
```

Each CLI run writes a copy of the terminal output to a timestamped log file:

```text
logs/aifb_log/YYYYMMDD_HHMMSS_run-all.log
logs/mutag_log/YYYYMMDD_HHMMSS_run-all.log
```

Use `--no-log` for temporary debugging runs where no log file is needed:

```bash
xai-mini --config configs/aifb.yaml --no-log analyze
```

If you did not install the package with `pip install -e .`, use:

```bash
PYTHONPATH=src python -m xai_miniproject.cli --config configs/aifb.yaml run-all
```

On Windows PowerShell, set `PYTHONPATH` like this:

```powershell
$env:PYTHONPATH="src"
python -m xai_miniproject.cli --config configs/aifb.yaml run-all
```

## Device Selection

The training device is configured in YAML:

```yaml
model:
  device: auto
```

`auto` chooses the first available backend in this order:

```text
CUDA -> MPS -> CPU
```

Use explicit values when you want reproducible hardware behavior:

```yaml
device: cpu       # always run on CPU
device: mps       # Apple Silicon GPU, only if torch.backends.mps.is_available()
device: cuda      # NVIDIA GPU
device: cuda:0    # specific NVIDIA GPU
```

GPU/MPS usually changes runtime, not the expected model quality. Metrics can still differ slightly because some GPU operations are less deterministic than CPU operations.

## Outputs

All generated outputs go to `artifacts/aifb/`:

```text
dataset_stats.json          # graph statistics for data analysis
metrics.json                # train/test accuracy and macro F1
predictions.csv             # GNN predictions for all target entities
model.pt                    # trained PyTorch checkpoint
aifb_without_labels.owl     # filtered ontology for concept learning
learning_problems.json      # positive/negative examples per predicted class
explanation_results.json    # CELOE or fallback explanations
explanation_results.csv     # flat explanation table
```

The config excludes AIFB `affiliation` and `employs` relations from the model graph and explanation ontology because they directly encode the target label.

## Switching Learners

Use CELOE:

```yaml
explanation:
  learner: CELOE
  backend: ontolearn
```

Try EvoLearner later by changing only the config:

```yaml
explanation:
  learner: EvoLearner
  backend: ontolearn
```

The Python wrapper looks up the learner class dynamically in Ontolearn.

## Switching Datasets

Copy `configs/mutag.template.yaml` to a new config file and fill in:

```text
rdf_path
rdf_format
train_path
test_path
all_labels_path
entity_column
label_column
target_node_type
exclude_predicates
label_names
```

The most important field is `exclude_predicates`: any predicate that directly reveals the label should be removed before training and explanation.

## Development Checks

```bash
python -m py_compile $(find src -name "*.py")
python -m pytest
```

On Windows PowerShell:

```powershell
Get-ChildItem -Recurse src -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
python -m pytest
```
