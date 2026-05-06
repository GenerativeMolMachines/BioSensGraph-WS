import gc
import numpy as np
import config
from smiles_utils import smi_canon_and_log
from io_utils import load_smiles, save_similarity
from fingerprints import generate_fingerprints
from similarity import run_similarity
import csv
from pathlib import Path


def main():
    df = load_smiles(config.in_path)

    bad_path = Path(config.out_path.parent, "invalid_smiles.csv")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    with bad_path.open("w", newline="", encoding="utf-8") as fb:
        bad_writer = csv.writer(fb)
        bad_writer.writerow(["id", "content"])

        df["content"] = [
            smi_canon_and_log(s, n, bad_writer)
            for s, n in zip(df["content"], df["id"])
        ]

    df = df.dropna(subset=["content"])
    print(f"Valid SMILES left: {len(df)}")
    print(f"Invalid SMILES saved to {bad_path}")

    fps, bit_counts, names, sequences = generate_fingerprints(df, config.radius, config.n_bits)
    del df
    gc.collect()

    bit_counts = np.array(bit_counts, dtype=np.int32)

    results = run_similarity(fps, bit_counts, names, sequences,
                             config.threshold, config.n_cpus)

    save_similarity(config.out_path, results)


if __name__ == "__main__":
    main()
