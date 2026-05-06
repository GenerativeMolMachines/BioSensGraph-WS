from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class EmbeddingConfig:
    csv_path: str
    name_col: str = "id"
    seq_col: str = "content"

    out_emb_path: str = str(ROOT / "data" / "data_protein" / "protein_embedding" / "esm2_emb.npy")
    out_ids_path: str = str(ROOT / "data" / "data_protein" / "protein_embedding" / "esm2_emb_ids.npy")

    model_name: str = "esm_2_t33_650M_UR50D"
    layer: int = 33
    device: str = "cuda"

    batch_size: int = 16
    log_every: int = 5000
