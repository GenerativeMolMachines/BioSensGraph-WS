from typing import Iterator, List, Tuple
import pandas as pd

def read_csv_sequences(path: str, name_col: str, seq_col: str) -> Tuple[List[str], List[str]]:
 """
 Reads ids and sequences from a CSV file.
 Assumes sequences are already cleaned and valid (AA alphabet).
 """
    df = pd.read_csv(path)
    ids = df[name_col].astype(str).tolist()
    seqs = df[seq_col].astype(str).tolist()
    return ids, seqs

def batch_iter(names: List[str], seqs: List[str], batch_size: int) -> Iterator[List[Tuple[str, str]]]:
 """
 Yields batches as list of (name, seq).
 """
    batch = []
    for n, s in zip(names, seqs):
        batch.append((n, s))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch