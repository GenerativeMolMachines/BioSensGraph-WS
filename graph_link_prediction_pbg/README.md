# BioSensGraph: Link Prediction

This project provides a reproducible workflow for biological knowledge graph link prediction using `PyTorch-BigGraph`.

## Purpose

- Learn entity and relation embeddings from graph triplets.
- Predict `interacts_with` links between molecular entities.
- Compare scoring setups (operators/comparators) through tracked experiments.

## Training Setup

- **Modeling framework:** `PyTorch-BigGraph`
- **Experiment tracking:** `DVC`
- **Execution environment:** local run or `SLURM` via scripts in `sbatch_tasks/`
- **Configuration sources:** `params.yaml`, `conf/`, `pbg_configs/`
- **Input format:** triplets `lhs  relation  rhs`

During training, the model optimizes scores of true edges against negative samples.

## Link Prediction

For a given `(lhs, relation)` query, the model ranks candidate `rhs` entities by score.  
Top-ranked entities are treated as the most likely interaction candidates.

## Quick Start

```bash
uv sync
uv pip install "dvc[s3]"
```

Prepare data and partitions:

```bash
./tasks/split.sh
./tasks/import_from_tsv.sh
```

Run an experiment locally:

```bash
uv run dvc exp run -S "operator=diagonal" -S "train.workers=16"
```

Run on SLURM:

```bash
sbatch sbatch_tasks/operator_diagonal.sh
```

## Outputs

- Checkpoints: `results_*/`
- Metrics: `metrics_*/`


Delete/recreate `dvc.lock` only if you intentionally want to rebuild lock state for the whole pipeline.
