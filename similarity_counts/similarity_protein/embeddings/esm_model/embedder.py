from typing import List, Tuple
import numpy as np
import torch

def _masked_mean_representations(reps: torch.Tensor, tokens: torch.Tensor, padding_idx: int) -> torch.Tensor:
 """
 reps: (B, T, C) float
 tokens: (B, T) long
 Returns: (B, C) masked mean over residue tokens only.

 Token layout in ESM2 batches:
 tokens[:, 0] = BOS
 tokens[:, 1:len(seq)+1] = residues
 tokens[:, len(seq)+1] = EOS
 rest = PAD
 """
    mask = tokens.ne(padding_idx)

    mask[:, 0] = False

    lengths = mask.sum(dim=1)  # counts tokens including EOS
    for i, L in enumerate(lengths.tolist()):
        if L > 0:
            mask[i, L - 1] = False

    mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (reps * mask_f).sum(dim=1)  # (B, C)
    denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
    return summed / denom

@torch.no_grad()
def embed_batch(
    model,
    alphabet,
    batch_converter,
    batch: List[Tuple[str, str]],
    layer: int,
    device: str,
) -> np.ndarray:
 """
 Compute per-protein embeddings for a batch.

 Returns float32 numpy array of shape (B, C).
 """
    labels, strs, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[layer], return_contacts=False)
    reps = out["representations"][layer]  # (B, T, C)

    emb = _masked_mean_representations(reps, tokens, alphabet.padding_idx)  # (B, C)
    return emb.cpu().numpy().astype("float32")
