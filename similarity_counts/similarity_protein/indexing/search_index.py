import numpy as np
import faiss
from config import *
from utils import load_embeddings, prepare_matrix


def search_all():
    faiss.omp_set_num_threads(N_CPUS)

    index = faiss.read_index(INDEX_PATH)
    index.hnsw.efSearch = EF_SEARCH

    emb, ids = load_embeddings(EMB_PATH, IDS_PATH, DIM)
    xq = prepare_matrix(emb, USE_COSINE)

    extra = 50
    D, I = index.search(xq, K + extra)

    n = I.shape[0]
    neighbors = np.empty((n, K), dtype=np.int64)
    scores = np.empty((n, K), dtype=np.float32)

    for i in range(n):
        mask = I[i] != i
        ii = I[i][mask][:K]
        dd = D[i][mask][:K]

        if ii.shape[0] < K:
            ii = I[i][:K]
            dd = D[i][:K]

        neighbors[i] = ii
        scores[i] = dd

    assert not np.any(neighbors == np.arange(n)[:, None]), "Self-match found!"

    return neighbors, scores, ids
