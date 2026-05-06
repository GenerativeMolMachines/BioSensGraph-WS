import faiss
from config import *
from utils import load_embeddings, prepare_matrix, ensure_dir_for_file


def build_index() -> None:
    faiss.omp_set_num_threads(N_CPUS)

    emb, _ = load_embeddings(EMB_PATH, IDS_PATH, DIM)
    xb = prepare_matrix(emb, USE_COSINE)

    metric = faiss.METRIC_INNER_PRODUCT if USE_COSINE else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(DIM, HNSW_M, metric)

    index.hnsw.efConstruction = EF_CONSTRUCT
    index.add(xb)

    ensure_dir_for_file(INDEX_PATH)
    faiss.write_index(index, INDEX_PATH)

    print("Index built and saved:", INDEX_PATH)
