# BioSensGraph: Predicting Biopolymer Interactions via Knowledge Graph Embedding on a Property Graph of Molecular Entities

## Overview
BioSensGraph is a research and engineering repository for molecular interaction modeling with knowledge graphs.

The project focuses on:
- integrating heterogeneous molecular data into a unified graph representation,
- learning entity/relation embeddings for link prediction,
- ranking candidate interactions such as `interacts_with`

The current pipeline is based on `PyTorch-BigGraph` in `graph_link_prediction_pbg/`, with parsing and data preparation components in `parsers/` and `data_preprocessing/`.

## Core Modules

This repository is organized as a modular workflow. Each module has a specific role in the end-to-end pipeline.

### 1) Data Collection Module (`parsers/`)

- Collects molecular entities and relations from heterogeneous sources (APIs, public datasets).
- Contains parser logic used to normalize raw inputs before graph construction.
- Includes parser-adjacent notebooks for exploratory parsing and validation.

### 2) Data Processing and Integration Module (`data_preprocessing/`)

- Cleans, merges, deduplicates, and harmonizes entities/relations into graph-ready formats.
- Prepares intermediate artifacts for downstream training and analytics.
- Stores the current preprocessing notebooks used for dataset assembly.

### 3) Link Prediction Module (`graph_link_prediction_pbg/`)

- Main production ML subsystem based on `PyTorch-BigGraph` (PBG).
- Uses transductive train/test splits prepared via `PyKEEN`, then imports them into PBG.
- Partitioned training, checkpointing, and metric evaluation.
- Provides reproducible training workflows with DVC and SLURM/local execution options.

### 4) Relation Augmentation Module (`similarity_counts/`)

- Builds additional similarity-based edges (sequence/structure similarity) used to enrich graph connectivity.
- Produces augmentation links.

## Project Structure
```
BioSensGraph-WS/
├── data_preprocessing/             # Notebook-based preprocessing workflows
├── parsers/                        # Source parsers and parsing notebooks
├── graph_link_prediction_pbg/      # Main link prediction system (PyTorch-BigGraph)
│   ├── conf/                       # Runtime/training configuration
│   ├── pbg_configs/                # PBG train/import config builders
│   ├── data_split/                 # Train/test split utilities
│   ├── tasks/                      # Pipeline task wrappers
│   ├── sbatch_tasks/               # SLURM launch scripts
│   ├── inference/                  # Inference and metric utilities
│   ├── resources/                  # Source TSV/CSV resources
│   ├── results_*/                  # Checkpoints/embeddings by operator setup
│   └── metrics_*/                  # Evaluation outputs by setup
└── similarity_counts/              # Similarity-edge generation (DNA/RNA/protein/etc.)
```

## Environment Strategy

Each major module is expected to use its own environment and dependency scope.

- `graph_link_prediction_pbg/` has its own ML-focused environment (link prediction).
- `similarity_counts/` has its own preprocessing-focused environment.
- Root-level parsing and preprocessing components (`parsers/`, `data_preprocessing/`) may use a separate lightweight environment.

## Link Prediction
Link prediction is the core ML task in this repository.  
Given `(lhs, relation)`, the model ranks candidate `rhs` entities and returns the most plausible missing edges.

## 1. Transductive Dataset Split

The graph is prepared using a **transductive split**, meaning:

- All **entities (nodes)** are present during training and testing.
- Only **edges (triples)** are divided into:
  - **80%** — training triples  
  - **20%** — test triples  

This setup evaluates the model’s ability to **recover missing interactions** among already known entities.

Triples are stored in TSV format: ```lhs_entity_id relation_type rhs_entity_id```

Current primary relations:

- `interacts_with`
- `has_similarity`

---

## 2. PyTorch-BigGraph (PBG) Training Pipeline

`graph_link_prediction_pbg/` is the main training pipeline for large graphs.

### 2.1 Graph Partitioning

PBG automatically partitions the graph:

- Supports distributed and parallel training.
- Ensures efficient negative sampling.
- Allows streaming of partitions instead of loading the whole graph into memory.

Each relation is parameterized by a trained **operator** (e.g., `diagonal`, `affine`, `complex_diagonal`, `translation`).

---

### 2.2 Optimization Objective

For each true triple `(h, r, t)`, the model computes a score:

score = f(h, r, t)

Higher scores correspond to plausible/true interactions.

Training uses ranking loss with negative sampling:

Loss = max(0, margin - score_positive + score_negative)

---

## 3. Negative Sampling

PBG uses **hybrid negative sampling**, combining both *batch-based* and *uniform* negatives.

### 3.1 Batch Negatives

These are generated from entity IDs inside the current minibatch.

### 3.2 Uniform Negatives

Sampled uniformly from **all entities** in the graph.


**Configuration used in this project:**

- **50** batch negatives
- **100** uniform negatives

Each positive triple is contrasted against 150 negatives.

---

## 4. What the Model Learns

### 4.1 Entity Embeddings

Dense vector representations ( **400 dimensions**) for:

- AA (peptides, proteins)   
- DNA
- RNA  
- Small Molecules  

These embeddings capture local and global graph structure.

---

### 4.2 Relation Operators

Operators transform a head entity embedding before comparison with the tail:

- **Diagonal** - elementwise scaling  
- **Affine** - linear transformation + bias  
- **ComplexDiagonal** - rotation in complex space  
- **Translation** - vector translation (TransE-style)  

Each relation learns its own operator parameters.

---

### 4.3 Comparator (Score Function)

Defines how similarity is computed between the transformed head embedding and the tail embedding:

- **Cosine similarity**
- **Dot product**
- **L2 distance**
- **Squared L2 distance**

---

## 5. Evaluation Protocol

As a ranking task, evaluation is based on:

- global graph-level evaluation:
**MRR**, **Hits@K**
- subset-specific evaluation for `interacts_with`:
**MRR**, **Hit@K**, **Precision@K**, **Recall@K**, **MAP**, **NDCG@K** 

Each test triple \(h, r, t) is evaluated by ranking:

- the true tail \(t) among **corrupted tails**, or  
- the true head \(h) among **corrupted heads**  

---

## 6. Inference Pipeline

The inference modules and notebooks provide:

1. loading trained embeddings and relation operators,
2. querying by entity and relation,
3. scoring candidates,
4. returning ranked predictions.

Typical outputs include:

- ranked `interacts_with` candidates,
- metric reports by operator/comparator,
- experiment-level comparisons across checkpoints.

---

## Database
The project utilizes [neo4j](https://neo4j.com/) graph database management system.

### Basic queries
[Documentation](https://neo4j.com/docs/cypher-cheat-sheet)

Count all records
```cypher
MATCH (n) RETURN COUNT(n)
```

Get statistics for all nodes with labels
```cypher
MATCH (n) RETURN DISTINCT LABELS(n), COUNT(n)
```

Get statistics for all nodes interactions
```cypher
MATCH (n)-[r:interacts_with]->(m) RETURN COUNT(r) AS relationship_count, LABELS(n) AS source_labels, LABELS(m) AS dest_labels ORDER BY relationship_count DESC 
```

## 7. Reproducible Run Procedure (PBG)

This section describes a practical run order for a new user in `graph_link_prediction_pbg/`.

### 7.1 Environment Setup

1. Install `uv` (if not installed yet):
   - [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
2. Go to the PBG module directory:
   - `cd BioSensGraph-WS/graph_link_prediction_pbg`
3. Sync all dependencies from `pyproject.toml`/`uv.lock`:
   - `uv sync`

### 7.2 Data Preparation

Place input files into `resources/` (project-local paths), then run split/import pipeline:

1. Build `train.tsv` and `test.tsv`:
   - `./tasks/split.sh`
2. Import TSV into PBG partitions:
   - `./tasks/import_from_tsv.sh`

### 7.3 Training

You can run training in two modes:

1. **SLURM mode (recommended for large runs)**
   - Run one operator at a time, for example:
   - `sbatch --cpus-per-task=120 --mem=20G --time=10:00:00 sbatch_tasks/operator_diagonal.sh`

2. **Local CPU mode**
   - Run task script directly:
   - `./sbatch_tasks/operator_diagonal.sh`
   - or explicitly:
   - `bash sbatch_tasks/operator_diagonal.sh`

After training, metric collection is launched by task scripts (for all configured comparators/epochs).

### 7.4 Recommended Experiment Tracking (DVC)

If you want full reproducibility of training states/checkpoints/metrics:

1. Run via DVC experiment command (already used in `sbatch_tasks/operator_*.sh`).
2. Save experiment explicitly:
   - `uv run dvc exp save -n <operator_name>`
3. Inspect/restore later:
   - `uv run dvc exp list`
   - `uv run dvc exp show`
   - `uv run dvc exp apply <exp_name>`


