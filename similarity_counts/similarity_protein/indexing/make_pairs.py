from pathlib import Path
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[2]
IDS_PATH = ROOT / "data" / "data_protein" / "protein_embedding" / "esm2_emb_ids.npy"
CSV_SRC  = ROOT / "data" / "data_protein" / "AA_len_1022.csv"
NEI_PATH = ROOT / "similarity_protein" / "indexing" / "faiss_index" / "neighbors_idx.npy"
SCR_PATH = ROOT / "similarity_protein" / "indexing" / "faiss_index" / "neighbors_scores.npy"

OUT_PATH = ROOT / "data" / "data_protein" / "pairs_for_alignment.csv"

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

_worker_ids = None
_worker_neighbors = None
_worker_scores = None


def _init_worker(ids_arr, neighbors_arr, scores_arr):
    global _worker_ids, _worker_neighbors, _worker_scores
    _worker_ids = ids_arr
    _worker_neighbors = neighbors_arr
    _worker_scores = scores_arr


def _build_chunk(args: tuple) -> pd.DataFrame:
 """Build edge DataFrame for rows [start_row, end_row)."""
    start_row, end_row = args
    ids_arr = _worker_ids
    neighbors = _worker_neighbors
    scores = _worker_scores
    N, K = neighbors.shape

    n_rows = end_row - start_row
    src_idx = np.repeat(np.arange(start_row, end_row, dtype=np.int64), K)
    dst_idx = neighbors[start_row:end_row].reshape(-1).astype(np.int64)
    sc = scores[start_row:end_row].reshape(-1)

    m = src_idx != dst_idx
    src_idx = src_idx[m]
    dst_idx = dst_idx[m]
    sc = sc[m]

    id1 = ids_arr[src_idx]
    id2 = ids_arr[dst_idx]

    df = pd.DataFrame({"id_1": id1, "id_2": id2, "score": sc})
    a, b = df["id_1"].values, df["id_2"].values
    df["_lo"] = np.minimum(a, b)
    df["_hi"] = np.maximum(a, b)
    return df


def main() -> None:
    global N_JOBS

    ids = np.load(str(IDS_PATH), allow_pickle=True).astype(str)
    neighbors = np.load(str(NEI_PATH), mmap_mode="r")
    scores = np.load(str(SCR_PATH), mmap_mode="r")

    N, K = neighbors.shape
    assert len(ids) == N, f"ids len {len(ids)} != N {N}"
    assert scores.shape == (N, K), f"scores shape {scores.shape} != {(N, K)}"

    df_src = pd.read_csv(str(CSV_SRC), usecols=["id", "content"])
    df_src["id"] = df_src["id"].astype(str)
    content_by_id = df_src.set_index("id")["content"]

    n_workers = min(N_JOBS, N)
    chunk_size = max(1, (N + n_workers - 1) // n_workers)
    ranges = []
    s = 0
    while s < N:
        e = min(s + chunk_size, N)
        ranges.append((s, e))
        s = e

    print(f"[INFO] building edges in parallel with {n_workers} workers, {len(ranges)} chunks")

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(ids, neighbors, scores),
    ) as pool:
        chunks = pool.map(_build_chunk, ranges, chunksize=1)

    df = pd.concat(chunks, ignore_index=True)

    df = (
        df.sort_values("score", ascending=False)
        .drop_duplicates(["_lo", "_hi"], keep="first")
        .drop(columns=["_lo", "_hi", "score"])
    )

    df["content_1"] = df["id_1"].map(content_by_id)
    df["content_2"] = df["id_2"].map(content_by_id)
    df = df[["id_1", "content_1", "id_2", "content_2"]]

    miss1 = int(df["content_1"].isna().sum())
    miss2 = int(df["content_2"].isna().sum())
    if miss1 or miss2:
        print(f"[WARN] missing content_1={miss1}, content_2={miss2} (id not found in {CSV_SRC})")

    outp = Path(OUT_PATH)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)

    print(f"[OK] undirected unique pairs saved: {len(df):,}")
    print(f"[OK] saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
