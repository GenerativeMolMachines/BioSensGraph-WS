import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from tqdm import tqdm

# Project root: .../graph_link_prediction_pbg
ROOT = Path(__file__).resolve().parents[1]

CKPT_REL = ROOT / "results_translation" / "l2_translation_checkpoint" / "epoch_22"
ENTITY_JSON = ROOT / "data" / "partitions" / "entity_names_molecules_0.json"

TRAIN_TSV = ROOT / "resources" / "train.tsv"
TEST_TSV = ROOT / "resources" / "test.tsv"

APTAMERS_CSV = ROOT / "resources" / "aptamers_dataset.csv"
ANTIBIOTICS_CSV = ROOT / "resources" / "antibiotic_small_molecules.csv"
ARGMOL_CSV = ROOT / "resources" / "argmol.csv"
AA_TRANSPORT_CSV = ROOT / "resources" / "aa_transport.csv"
AA_CANCER_CSV = ROOT / "resources" / "aa_cancer.csv"
AA_SIG_BIO_CSV = ROOT / "resources" / "aa_sig_bio.csv"

with open(CKPT_REL / "checkpoint_version.txt", "r") as f:
    CKPT_VER = int(f.read().strip())

EMB_H5 = CKPT_REL / f"embeddings_molecules_0.v{CKPT_VER}.h5"
MODEL_H5 = CKPT_REL / f"model.v{CKPT_VER}.h5"

DIM = 400
RELATION_NAME = "interacts_with"
RELATION_IDX = 0  # to be sure that interacts_with == 0
NUM_NEGATIVES_PER_TRUE = 150
K_VALUES = [1, 3, 5, 10, 25, 50, 75, 100]

DEVICE = "cpu"
NCPU = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

OUTPUT_DIR = ROOT / "metrics" / "inference_translation_l2_notebook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_GLOBAL_SUBSET_NODES_DICT = None
_GLOBAL_ENTITY_TO_IDX = None
_GLOBAL_ALL_EMB = None
_GLOBAL_REL_VEC = None
_GLOBAL_ALL_NEIGHBORS = None
_GLOBAL_NUM_NEGATIVES_PER_TRUE = None
_GLOBAL_K_VALUES = None


def load_entities(path: Path):
    with open(path, "rt") as f:
        names = json.load(f)
    return names, {e: i for i, e in enumerate(names)}


def load_embeddings(path: Path, device: str = "cpu") -> torch.Tensor:
    with h5py.File(path, "r") as hf:
        emb = torch.from_numpy(hf["embeddings"][...]).float()
    return emb.to(device)


def load_translation_vector(model_h5_path: Path, relation_idx: int, dim: int) -> torch.Tensor:
    candidates = [
        f"model/relations/{relation_idx}/operator/rhs/translation",
        f"model/relations/{relation_idx}/operator/rhs/bias",
        f"model/relations/{relation_idx}/operator/rhs/vec",
    ]

    with h5py.File(model_h5_path, "r") as hf:
        for key in candidates:
            if key in hf:
                vec = torch.from_numpy(hf[key][...]).float()
                if tuple(vec.shape) == (dim,):
                    return vec

        found = None

        def visit(name, obj):
            nonlocal found
            if found is not None:
                return
            if isinstance(obj, h5py.Dataset):
                if f"model/relations/{relation_idx}/operator/rhs/" in name and tuple(obj.shape) == (dim,):
                    found = torch.from_numpy(obj[...]).float()

        hf.visititems(visit)

    if found is None:
        raise RuntimeError(f"Translation vector not found for relation_idx={relation_idx}")

    return found


def read_relation_tsv(tsv_path: Path, relation_name: str = RELATION_NAME) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["source", "relation", "target"])
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df = df[df["relation"] == relation_name].copy()
    return df


def load_pairs_symmetric(tsv_path: Path, relation_name: str = RELATION_NAME):
    df = read_relation_tsv(tsv_path, relation_name=relation_name)
    pairs = []
    for _, row in df.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        pairs.append((s, t))
        pairs.append((t, s))
    return pairs


def build_nodes_dict_from_df_symmetric(df: pd.DataFrame) -> dict:
    d = defaultdict(set)
    for _, row in df.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        d[s].add(t)
        d[t].add(s)
    return d


def load_all_neighbors_symmetric(*tsv_paths: Path) -> dict:
    neigh = defaultdict(set)
    for path in tsv_paths:
        if not Path(path).is_file():
            continue
        for s, t in load_pairs_symmetric(Path(path), relation_name=RELATION_NAME):
            neigh[s].add(t)
    return neigh



def score_translation_l2(lhs: torch.Tensor, rhs: torch.Tensor, rel_vec: torch.Tensor) -> torch.Tensor:
    rhs_t = rhs + rel_vec
    return -torch.norm(lhs - rhs_t, p=2, dim=-1)


def sample_negatives(
    source_id: str,
    all_neighbors: dict,
    entity_to_idx: dict,
    num_nodes: int,
    num_negatives: int,
    rng: random.Random,
):
    src_idx = entity_to_idx.get(source_id)
    if src_idx is None:
        return []

    excluded = {src_idx}
    for n_id in all_neighbors.get(source_id, set()):
        j = entity_to_idx.get(n_id)
        if j is not None:
            excluded.add(j)

    avail = [i for i in range(num_nodes) if i not in excluded]
    if not avail:
        return []

    if len(avail) <= num_negatives:
        return avail

    return rng.sample(avail, num_negatives)


def evaluate_source_translation_l2(
    source_id: str,
    true_targets: set,
    entity_to_idx: dict,
    all_emb: torch.Tensor,
    rel_vec: torch.Tensor,
    all_neighbors: dict,
    num_negatives_per_true: int,
    k_values,
    rng: random.Random,
):
    src_idx = entity_to_idx.get(source_id)
    if src_idx is None:
        return None

    true_indices = sorted({entity_to_idx[t] for t in true_targets if t in entity_to_idx})
    if not true_indices:
        return None

    n_true = len(true_indices)
    num_neg = num_negatives_per_true * n_true

    neg_indices = sample_negatives(
        source_id=source_id,
        all_neighbors=all_neighbors,
        entity_to_idx=entity_to_idx,
        num_nodes=all_emb.shape[0],
        num_negatives=num_neg,
        rng=rng,
    )
    if not neg_indices:
        return None

    cand = true_indices + neg_indices
    labels = np.zeros(len(cand), dtype=np.int32)
    labels[:n_true] = 1

    lhs = all_emb[src_idx].view(1, -1).expand(len(cand), -1)
    rhs = all_emb[cand]
    rel = rel_vec.view(1, -1).expand(len(cand), -1)
    scores = score_translation_l2(lhs, rhs, rel).cpu().numpy()

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]

    rank_list = [int(np.where(order == i)[0][0] + 1) for i in range(n_true)]

    metrics = {
        "source_id": source_id,
        "num_true": n_true,
        "num_negatives": len(neg_indices),
        "num_candidates": len(cand),
        "mean_rank": float(np.mean(rank_list)),
        "median_rank": float(np.median(rank_list)),
        "best_rank": int(np.min(rank_list)),
        "worst_rank": int(np.max(rank_list)),
        "mrr": float(np.mean([1.0 / r for r in rank_list])),
        "ranks": rank_list,
        "scores": scores.astype(float).tolist(),
        "labels": labels.astype(int).tolist(),
    }

    ap = 0.0
    correct = 0
    for i, lab in enumerate(sorted_labels, start=1):
        if lab == 1:
            correct += 1
            ap += correct / i
    metrics["map"] = float(ap / n_true)

    for k in k_values:
        k_eff = min(k, len(cand))
        topk = sorted_labels[:k_eff]
        tp = int(np.sum(topk))

        metrics[f"hit@{k}"] = float(1.0 if tp > 0 else 0.0)
        metrics[f"precision@{k}"] = float(tp / k_eff if k_eff else 0.0)
        metrics[f"recall@{k}"] = float(tp / n_true if n_true else 0.0)

        p = metrics[f"precision@{k}"]
        r = metrics[f"recall@{k}"]
        metrics[f"f1@{k}"] = float(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

        dcg = float(np.sum(topk / np.log2(np.arange(2, k_eff + 2))))
        ideal = np.concatenate([np.ones(n_true), np.zeros(len(cand) - n_true)])[:k_eff]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, k_eff + 2))))
        metrics[f"ndcg@{k}"] = float(dcg / idcg) if idcg > 0 else 0.0

    return metrics


def init_pool_for_subset(
    subset_nodes_dict,
    entity_to_idx,
    all_emb,
    rel_vec,
    all_neighbors,
    num_negatives_per_true,
    k_values,
):
    global _GLOBAL_SUBSET_NODES_DICT
    global _GLOBAL_ENTITY_TO_IDX
    global _GLOBAL_ALL_EMB
    global _GLOBAL_REL_VEC
    global _GLOBAL_ALL_NEIGHBORS
    global _GLOBAL_NUM_NEGATIVES_PER_TRUE
    global _GLOBAL_K_VALUES

    _GLOBAL_SUBSET_NODES_DICT = subset_nodes_dict
    _GLOBAL_ENTITY_TO_IDX = entity_to_idx
    _GLOBAL_ALL_EMB = all_emb
    _GLOBAL_REL_VEC = rel_vec
    _GLOBAL_ALL_NEIGHBORS = all_neighbors
    _GLOBAL_NUM_NEGATIVES_PER_TRUE = num_negatives_per_true
    _GLOBAL_K_VALUES = k_values


def eval_source_subset(packed):
    sid, rng_seed = packed
    return evaluate_source_translation_l2(
        source_id=sid,
        true_targets=_GLOBAL_SUBSET_NODES_DICT[sid],
        entity_to_idx=_GLOBAL_ENTITY_TO_IDX,
        all_emb=_GLOBAL_ALL_EMB,
        rel_vec=_GLOBAL_REL_VEC,
        all_neighbors=_GLOBAL_ALL_NEIGHBORS,
        num_negatives_per_true=_GLOBAL_NUM_NEGATIVES_PER_TRUE,
        k_values=_GLOBAL_K_VALUES,
        rng=random.Random(rng_seed),
    )


def pooled_scores_labels(results_df: pd.DataFrame):
    all_scores, all_labels = [], []
    for _, row in results_df.iterrows():
        if "scores" in row and "labels" in row and row["scores"] is not None:
            all_scores.extend(row["scores"])
            all_labels.extend(row["labels"])
    if not all_scores:
        return None, None
    return np.asarray(all_scores, dtype=np.float64), np.asarray(all_labels, dtype=np.int64)


def binary_metrics_from_pooled(all_scores: np.ndarray, all_labels: np.ndarray) -> dict:
    out: dict[str, float | None] = {
        "bin_threshold": None,
        "bin_precision": None,
        "bin_recall": None,
        "bin_f1": None,
    }

    prec, rec, thr = precision_recall_curve(all_labels, all_scores)
    f1_vals = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = int(np.argmax(f1_vals))

    if len(thr) == 0:
        return out

    best_thr = float(thr[best_idx]) if best_idx < len(thr) else float(thr[-1])
    y_pred = (all_scores >= best_thr).astype(int)

    out["bin_threshold"] = best_thr
    out["bin_precision"] = float(precision_score(all_labels, y_pred, zero_division=0))
    out["bin_recall"] = float(recall_score(all_labels, y_pred, zero_division=0))
    out["bin_f1"] = float(f1_score(all_labels, y_pred, zero_division=0))
    return out


def summary_row_for_subset(name: str, df_subset: pd.DataFrame, results_df: pd.DataFrame, binary: dict) -> dict:
    unique_sources = results_df["source_id"].nunique()
    subset_num_edges = int(len(df_subset))
    subset_num_unique_nodes = int(pd.concat([df_subset["source"], df_subset["target"]]).nunique())

    row = {
        "subset": name,
        "subset_num_edges": subset_num_edges,
        "subset_num_unique_nodes": subset_num_unique_nodes,
        "num_evaluated_sources": int(unique_sources),
        "avg_links_per_node": float(results_df["num_true"].mean()),
        "mean_rank": float(results_df["mean_rank"].mean()),
        "median_rank": float(results_df["median_rank"].median()),
        "map": float(results_df["map"].mean()),
        "mrr": float(results_df["mrr"].mean()),
    }

    for k in K_VALUES:
        for col_prefix in ("hit", "recall", "precision", "ndcg"):
            c = f"{col_prefix}@{k}"
            if c in results_df.columns:
                row[c] = float(results_df[c].mean())

    row.update(binary)
    return row


def run_subset(
    name: str,
    df_subset: pd.DataFrame,
    entity_to_idx: dict,
    all_emb: torch.Tensor,
    rel_vec: torch.Tensor,
    all_neighbors: dict,
):
    subset_nodes_dict = build_nodes_dict_from_df_symmetric(df_subset)
    sources = sorted([sid for sid, tgts in subset_nodes_dict.items() if len(tgts) > 0])

    if not sources:
        print(f"[WARN] No sources found for subset: {name}")
        return None

    rows = []
    with Pool(
        processes=NCPU,
        initializer=init_pool_for_subset,
        initargs=(
            subset_nodes_dict,
            entity_to_idx,
            all_emb,
            rel_vec,
            all_neighbors,
            NUM_NEGATIVES_PER_TRUE,
            K_VALUES,
        ),
    ) as pool:
        iterator = pool.imap(eval_source_subset, ((sid, SEED + i) for i, sid in enumerate(sources)))
        for m in tqdm(iterator, total=len(sources), desc=name):
            if m is not None:
                rows.append(m)

    if not rows:
        print(f"[WARN] No evaluation rows produced for subset: {name}")
        return None

    results_df = pd.DataFrame(rows)

    pooled_s, pooled_l = pooled_scores_labels(results_df)
    binary = (
        binary_metrics_from_pooled(pooled_s, pooled_l)
        if pooled_s is not None
        else {"bin_threshold": None, "bin_precision": None, "bin_recall": None, "bin_f1": None}
    )

    export_df = results_df.copy()
    export_df["ranks"] = export_df["ranks"].apply(json.dumps)
    export_df = export_df.drop(columns=["scores", "labels"], errors="ignore")
    export_df.to_csv(OUTPUT_DIR / f"{name}_translation_l2_e22.csv", index=False)

    np.savez_compressed(
        OUTPUT_DIR / f"{name}_translation_l2_e22_pooled.npz",
        scores=pooled_s if pooled_s is not None else np.array([]),
        labels=pooled_l if pooled_l is not None else np.array([]),
    )

    return summary_row_for_subset(name, df_subset, results_df, binary)


def main():
    print("Loading train/test relation splits...")
    train_iw = read_relation_tsv(TRAIN_TSV, relation_name=RELATION_NAME)
    test_iw = read_relation_tsv(TEST_TSV, relation_name=RELATION_NAME)

    print("Loading subset entity lists...")
    aptamers = pd.read_csv(APTAMERS_CSV)
    antibiotics = pd.read_csv(ANTIBIOTICS_CSV)
    argmol = pd.read_csv(ARGMOL_CSV)
    aa_transport = pd.read_csv(AA_TRANSPORT_CSV)
    aa_cancer = pd.read_csv(AA_CANCER_CSV)
    aa_sig_bio = pd.read_csv(AA_SIG_BIO_CSV)

    aid = set(aptamers["id"].astype(str))
    bid = set(antibiotics["id"].astype(str))
    pid = set(argmol["id"].astype(str))
    tid = set(aa_transport["id"].astype(str))
    cid = set(aa_cancer["id"].astype(str))
    sid = set(aa_sig_bio["id"].astype(str))

    rows_aptamer = test_iw[test_iw["source"].isin(aid) | test_iw["target"].isin(aid)].copy()
    rows_antibiotics = test_iw[test_iw["source"].isin(bid) | test_iw["target"].isin(bid)].copy()
    rows_argmol = test_iw[test_iw["source"].isin(pid) | test_iw["target"].isin(pid)].copy()
    rows_aa_transport = test_iw[test_iw["source"].isin(tid) | test_iw["target"].isin(tid)].copy()
    rows_aa_cancer = test_iw[test_iw["source"].isin(cid) | test_iw["target"].isin(cid)].copy()
    rows_aa_sig_bio = test_iw[test_iw["source"].isin(sid) | test_iw["target"].isin(sid)].copy()

    print("Loading entity dictionary and embeddings...")
    _, entity_to_idx = load_entities(ENTITY_JSON)
    all_emb = load_embeddings(EMB_H5, device=DEVICE)
    rel_vec = load_translation_vector(MODEL_H5, relation_idx=RELATION_IDX, dim=DIM).to(DEVICE)

    print("Building filtered known-neighbor index from train+test...")
    all_neighbors = load_all_neighbors_symmetric(TRAIN_TSV, TEST_TSV)

    summary_rows = []

    for subset_name, df_sub in (
        ("aptamers", rows_aptamer),
        ("antibiotics", rows_antibiotics),
        ("argmol", rows_argmol),
        ("aa_transport", rows_aa_transport),
        ("aa_cancer", rows_aa_cancer),
        ("aa_sig_bio", rows_aa_sig_bio),
    ):
        print(f"\nRunning subset: {subset_name}")
        print(f"edges: {len(df_sub)}")
        print(f"unique nodes: {pd.concat([df_sub['source'], df_sub['target']]).nunique() if len(df_sub) else 0}")

        srow = run_subset(
            name=subset_name,
            df_subset=df_sub,
            entity_to_idx=entity_to_idx,
            all_emb=all_emb,
            rel_vec=rel_vec,
            all_neighbors=all_neighbors,
        )
        if srow is not None:
            summary_rows.append(srow)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUTPUT_DIR / "summary_translation_l2_e22.csv", index=False)
        print("\nSaved summary:")
        print(summary_df)
    else:
        print("\nNo summary rows produced")


if __name__ == "__main__":
    main()