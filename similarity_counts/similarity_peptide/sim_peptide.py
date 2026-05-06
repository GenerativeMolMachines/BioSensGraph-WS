import csv
from itertools import combinations
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from aligment import calculate_alignment_similarity


def work_on_pair(args):
 """Function for parallel processing pair sequences"""
    id1, seq1, id2, seq2 = args
    sim = calculate_alignment_similarity(
        sequence1=seq1,
        sequence2=seq2,
        alignment_type="global"
    )
    return id1, seq1, id2, seq2, sim


def main():
    freeze_support()

    root = Path(__file__).resolve().parents[1]
    input_csv = root / "data" / "data_peptide" / "AA_len_100.csv"
    output_csv = root / "data" / "data_peptide" / "AA_len_100_similarity.csv"
    n_cores = cpu_count()

    df = pd.read_csv(str(input_csv))

    pairs = list(combinations(df.itertuples(index=False), 2))
    total_pairs = len(pairs)

    pair_iter = [(p1.id, p1.content, p2.id, p2.content) for p1, p2 in pairs]

    with Pool(processes=n_cores) as pool, open(str(output_csv), "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["id_1", "sequence_1", "id_2", "sequence_2", "alignment"])

        for id1, seq1, id2, seq2, sim in tqdm(
            pool.imap_unordered(work_on_pair, pair_iter),
            total=total_pairs,
            desc="Calculating Peptide alignments"
        ):
            writer.writerow([id1, seq1, id2, seq2, sim])


if __name__ == "__main__":
    main()
