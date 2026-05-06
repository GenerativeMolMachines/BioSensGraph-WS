import os
import numpy as np
import faiss


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_embeddings(emb_path: str, ids_path: str, dim: int):
    ids = np.load(ids_path, allow_pickle=True)
    emb = np.load(emb_path, mmap_mode="r")  # correct way for .npy

    if emb.ndim != 2 or emb.shape[1] != dim:
        raise ValueError(f"emb shape {emb.shape}, expected (*,{dim})")
    if len(ids) != emb.shape[0]:
        raise ValueError(f"ids len {len(ids)} != N {emb.shape[0]}")
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)

    return emb, ids


def prepare_matrix(emb: np.ndarray, use_cosine: bool) -> np.ndarray:
    x = np.array(emb, dtype=np.float32, copy=True)
    if use_cosine:
        faiss.normalize_L2(x)
    return x