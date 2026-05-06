import csv
import os
from itertools import combinations, islice
from pathlib import Path

from multiprocessing import Pool, cpu_count, freeze_support

import pandas as pd
from tqdm import tqdm

from aligment import calculate_alignment_similarity

SAVE_EVERY_N = 100_000
IMAP_CHUNK_SIZE = 200
MAX_WORKERS = 128


def work_on_pair(args):
 """Function for parallel processing pair sequences."""
    id1, seq1, id2, seq2 = args
    sim = calculate_alignment_similarity(
        sequence1=seq1,
        sequence2=seq2,
        alignment_type="global",
    )
    return id1, seq1, id2, seq2, sim


def iter_pairs_skip(df, start_index):
 """Iterator for pairs (id1, content1, id2, content2), propuskaet first start_index pairs."""
    rows = list(df.itertuples(index=False))  # single list rows, order combinations deterministic
    for p1, p2 in islice(combinations(rows, 2), start_index, None):
        yield (p1.id, p1.content, p2.id, p2.content)


def main():
    freeze_support()

    root = Path(__file__).resolve().parents[2]
    input_csv = root / "data" / "data_protein" / "aa_q4.csv"
    output_csv = root / "data" / "data_protein" / "aa_q4_similarity.csv"

    n_cores = cpu_count()
    print(f"CPU: {n_cores}")

    df = pd.read_csv(str(input_csv))
    n = len(df)
    total_pairs = n * (n - 1) // 2

    start_index = 0
    file_exists = os.path.isfile(output_csv)
    if file_exists:
        with open(output_csv, "r", newline="") as f_to:
            start_index = sum(1 for _ in f_to) - 1
        if start_index < 0:
            start_index = 0
        if start_index > 0:
            print(f"Resuming pair {start_index + 1} (already processed {start_index} pairs)")

    to_process = total_pairs - start_index
    if to_process <= 0:
        print("All pair already processed.")
        return

    pairs_iter = iter_pairs_skip(df, start_index)
    mode = "a" if file_exists else "w"

    with Pool(processes=n_cores) as pool, open(str(output_csv), mode, newline="") as f_out:
        writer = csv.writer(f_out)
        if not file_exists:
            writer.writerow(["id_1", "sequence_1", "id_2", "sequence_2", "alignment"])

        written = 0
        for id1, seq1, id2, seq2, sim in tqdm(
            pool.imap_unordered(work_on_pair, pairs_iter, chunksize=IMAP_CHUNK_SIZE),
            total=to_process,
            desc="Calculating Protein alignments",
        ):
            writer.writerow([id1, seq1, id2, seq2, sim])
            written += 1
            if written % SAVE_EVERY_N == 0:
                f_out.flush()
                os.fsync(f_out.fileno())


if __name__ == "__main__":
    main()
