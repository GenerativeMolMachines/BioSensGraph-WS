import os
import numpy as np

def create_memmap(path: str, shape, dtype="float32") -> np.memmap:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)

def flush_memmap(mm: np.memmap) -> None:
    mm.flush()
