import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMB_PATH = str(ROOT / "data" / "data_protein" / "protein_embedding" / "esm2_emb.npy")
IDS_PATH = str(ROOT / "data" / "data_protein" / "protein_embedding" / "esm2_emb_ids.npy")

INDEX_PATH = str(ROOT / "similarity_protein" / "indexing" / "faiss_index" / "proteins_hnsw.index")
NEIGHBORS_PATH = str(ROOT / "similarity_protein" / "indexing" / "faiss_index" / "neighbors_idx.npy")
SCORES_PATH = str(ROOT / "similarity_protein" / "indexing" / "faiss_index" / "neighbors_scores.npy")

DIM = 1280
USE_COSINE = True
K = 200

HNSW_M = 48
EF_CONSTRUCT = 400
EF_SEARCH = 256

N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
