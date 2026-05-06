from rdkit import DataStructs
from collections import defaultdict
import math, numpy as np
from tqdm import tqdm
import multiprocessing as mp


def build_buckets(bit_counts):
    buckets = defaultdict(list)
    for idx, bc in enumerate(bit_counts):
        buckets[int(bc)].append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}


def process_one(i, bit_counts, fps, names, sequences, buckets, threshold):
    ai = bit_counts[i]
    min_b = math.ceil(threshold * ai)
    max_b = math.floor(ai / threshold)

    js = []
    for bc in range(min_b, max_b + 1):
        js.extend(buckets.get(bc, []))
    js = [j for j in js if j > i]
    if not js:
        return []

    sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in js])
    return [
        (names[i], sequences[i], names[j], sequences[j], f"{sim:.6f}")
        for j, sim in zip(js, sims)
        if sim >= threshold
    ]


def run_similarity(fps, bit_counts, names, sequences, threshold, n_cpus=4):
    buckets = build_buckets(bit_counts)
    n = len(fps)
    print(f"Starting similarity search on {n_cpus} cores for {n} molecules...")

    with mp.Pool(processes=n_cpus) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_one,
                    [(i, bit_counts, fps, names, sequences, buckets, threshold) for i in range(n)],
                ),
                total=n,
                desc="Similarity search (parallel)",
            )
        )

    flat = [r for chunk in results for r in chunk]
    return flat
