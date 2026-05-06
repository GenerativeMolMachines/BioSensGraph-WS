import os
import numpy as np

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_names(path: str, names) -> None:
    ensure_dir_for_file(path)
    np.save(path, np.array(names, dtype=object))
