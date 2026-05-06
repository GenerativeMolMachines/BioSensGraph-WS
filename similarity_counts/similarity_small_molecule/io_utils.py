import pandas as pd
from pathlib import Path
import csv

def load_smiles(in_path: Path) -> pd.DataFrame:
    df = pd.read_csv(in_path, usecols=["content", "id"])
    print(f"Loaded {len(df)} SMILES")
    return df

def save_similarity(out_path: Path, data):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id_1", "content_1", "id_2", "content_2", "tanimoto"])
        for row in data:
            writer.writerow(row)
    print(f"Saved results to {out_path}")
