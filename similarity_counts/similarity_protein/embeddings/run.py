import os
from pathlib import Path
import numpy as np

from config import EmbeddingConfig
from esm_model.esm_model import load_esm_model
from esm_model.esm_loader import read_csv_sequences, batch_iter
from esm_model.embedder import embed_batch
from utils.io import save_names, ensure_dir_for_file
from utils.checkpoint import create_memmap, flush_memmap


DIM = 1280 


def run(cfg: EmbeddingConfig) -> None:
    names, seqs = read_csv_sequences(cfg.csv_path, cfg.name_col, cfg.seq_col)
    n = len(seqs)

    model, alphabet, batch_converter = load_esm_model(cfg.model_name, cfg.device)

    raw_path = cfg.out_emb_path + ".raw"
    ensure_dir_for_file(raw_path)
    mm = create_memmap(raw_path, shape=(n, DIM), dtype="float32")

    idx = 0
    for batch in batch_iter(names, seqs, cfg.batch_size):
        emb = embed_batch(model, alphabet, batch_converter, batch, cfg.layer, cfg.device)

        if emb.ndim != 2 or emb.shape[1] != DIM:
            raise ValueError(f"bad emb shape {emb.shape}, expected (B,{DIM})")

        b = emb.shape[0]
        mm[idx:idx + b] = emb
        idx += b

        if cfg.log_every and (idx % cfg.log_every == 0):
            flush_memmap(mm)
            print(f"[INFO] embedded {idx}/{n}")

    flush_memmap(mm)

    if idx != n:
        raise ValueError(f"idx != n: {idx} != {n}")

    arr = np.array(mm, dtype=np.float32, copy=True)
    del mm  # close memmap before overwriting/cleaning if needed

    ensure_dir_for_file(cfg.out_emb_path)
    np.save(cfg.out_emb_path, arr)

    save_names(cfg.out_ids_path, names)

    chk = np.load(cfg.out_emb_path, mmap_mode="r")
    if chk.dtype != np.float32 or chk.shape != (n, DIM):
        raise ValueError(f"saved emb invalid: dtype={chk.dtype} shape={chk.shape}")

    print(f"[DONE] embeddings saved: {cfg.out_emb_path} shape={chk.shape} dtype={chk.dtype}")
    print(f"[DONE] ids saved: {cfg.out_ids_path} count={n}")
    print(f"[NOTE] temp raw left on disk: {raw_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cfg = EmbeddingConfig(
        csv_path=str(root / "data" / "data_protein" / "AA_len_1022.csv"),
        out_emb_path=str(root / "data" / "data_protein" / "protein_embedding" / "esm2_emb.npy"),
        out_ids_path=str(root / "data" / "data_protein" / "protein_embedding" / "esm2_emb_ids.npy"),
        batch_size=16,
        layer=33,
        device="cuda",
        log_every=5000,
    )
    run(cfg)
