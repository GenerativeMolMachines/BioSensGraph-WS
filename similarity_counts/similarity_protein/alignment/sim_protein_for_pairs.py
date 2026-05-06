import csv
import os
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from aligment import calculate_alignment_similarity

SAVE_EVERY_N = 100_000
CHUNK_SIZE = 100_000
IMAP_CHUNK_SIZE = 200  # batch for passing to workers — less IPC
MAX_WORKERS = 128


def work_on_pair(args):
    id1, seq1, id2, seq2 = args
    sim = calculate_alignment_similarity(
        sequence1=seq1,
        sequence2=seq2,
        alignment_type="global",
    )
    return id1, seq1, id2, seq2, sim


def _count_rows(input_csv: str) -> int:
    total = 0
    for chunk in pd.read_csv(
        input_csv, sep=None, engine="python", chunksize=CHUNK_SIZE, usecols=["id_1"]
    ):
        total += len(chunk)
    return total


def _iter_pairs_from_index(input_csv: str, start_index: int):
    global_idx = 0
    for chunk in pd.read_csv(
        input_csv, sep=None, engine="python", chunksize=CHUNK_SIZE
    ):
        for _, row in chunk.iterrows():
            if global_idx >= start_index:
                yield (
                    row["id_1"],
                    row["content_1"],
                    row["id_2"],
                    row["content_2"],
                )
            global_idx += 1


def main():
    freeze_support()

    root = Path(__file__).resolve().parents[2]
    input_csv = root / "data" / "data_protein" / "pairs_for_alignment.csv"
    output_csv = root / "data" / "data_protein" / "pairs_for_alignment_similarity.csv"
    required = ["id_1", "content_1", "id_2", "content_2"]

    first_chunk = pd.read_csv(str(input_csv), sep=None, engine="python", nrows=1)
    for col in required:
        if col not in first_chunk.columns:
            raise ValueError(f"In input file is missing column: {col}")
    del first_chunk

    n_cores = min(MAX_WORKERS, cpu_count())
    print(f"CPU: {n_cores}")

    start_index = 0
    file_exists = os.path.isfile(output_csv)
    if file_exists:
        with open(output_csv, "r", newline="") as f_to:
            start_index = sum(1 for _ in f_to) - 1
        if start_index < 0:
            start_index = 0
        if start_index > 0:
            print(f"Resuming pair {start_index + 1} (already processed {start_index} pairs)")

    total_pairs = _count_rows(str(input_csv))
    to_process = total_pairs - start_index
    if to_process <= 0:
        print("All pairs already processed")
        return

    pairs_iter = _iter_pairs_from_index(str(input_csv), start_index)

    mode = "a" if file_exists else "w"
    with Pool(processes=n_cores) as pool, open(str(output_csv), mode, newline="") as f_out:
        writer = csv.writer(f_out)
        if not file_exists:
            writer.writerow(["id_1", "content_1", "id_2", "content_2", "alignment"])

        written = 0
        for id1, seq1, id2, seq2, sim in tqdm(
            pool.imap(work_on_pair, pairs_iter, chunksize=IMAP_CHUNK_SIZE),
            total=to_process,
            desc="Calculating Protein alignments for pairs",
        ):
            writer.writerow([id1, seq1, id2, seq2, sim])
            written += 1
            if written % SAVE_EVERY_N == 0:
                f_out.flush()
                os.fsync(f_out.fileno())


if __name__ == "__main__":
    main()
