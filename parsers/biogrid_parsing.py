import pandas as pd
import requests
import time
from tqdm import tqdm
from pathlib import Path

# === CONFIG ===
INPUT_FILE = Path("../../data/BIOGRID-ALL-5.0.250.tab3.txt")
OUTPUT_DIR = Path("../../data/biogrid_processed")
OUTPUT_DIR.mkdir(exist_ok=True)

FILTERED_FILE = OUTPUT_DIR / "biogrid_filtered.csv"
FINAL_FILE = OUTPUT_DIR / "biogrid_ppi_unique.csv"
PROTEIN_FILE = OUTPUT_DIR / "protein_data_biogrid.csv"
INTERACTION_FILE = OUTPUT_DIR / "protein_protein_interaction_data.csv"
STATS_FILE = OUTPUT_DIR / "biogrid_stats.txt"
AMBIGUITY_TXT = OUTPUT_DIR / "biogrid_ambiguity_stats.txt"
NAME_TO_SEQ = OUTPUT_DIR / "name_to_sequences.csv"
SEQ_TO_NAME = OUTPUT_DIR / "sequence_to_names.csv"

# === LOAD DATA ===
print(f"[INFO] Loading dataset from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str, low_memory=False)
print(f"[INFO] Loaded: {len(df):,} rows, {len(df.columns)} columns.")

# === KEEP REQUIRED COLUMNS ===
COLUMNS_TO_KEEP = [
    "SWISS-PROT Accessions Interactor A",
    "SWISS-PROT Accessions Interactor B",
    "Score",
    "Qualifications",
]
missing_cols = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
if missing_cols:
    raise ValueError(f"[ERROR] Missing columns in file: {missing_cols}")

df = df[COLUMNS_TO_KEEP].copy()
print(f"[INFO] Columns selected. Remaining shape: {df.shape}")

# === REMOVE INVALID AND AMBIGUOUS ROWS ===
initial_len = len(df)
df = df.dropna(subset=["SWISS-PROT Accessions Interactor A", "SWISS-PROT Accessions Interactor B"])
df = df[
    (df["SWISS-PROT Accessions Interactor A"] != "-") &
    (df["SWISS-PROT Accessions Interactor B"] != "-")
]
df = df[
    ~df["SWISS-PROT Accessions Interactor A"].str.contains(r"\|", na=False) &
    ~df["SWISS-PROT Accessions Interactor B"].str.contains(r"\|", na=False)
]
df = df.drop_duplicates(subset=["SWISS-PROT Accessions Interactor A", "SWISS-PROT Accessions Interactor B"]).reset_index(drop=True)
filtered_len = len(df)
print(f"[INFO] Filtered out ambiguous/invalid rows: {initial_len - filtered_len:,} removed, {filtered_len:,} remain.")

# === SAVE FILTERED DATA ===
df.to_csv(FILTERED_FILE, index=False)
print(f"[INFO] Filtered dataset saved to {FILTERED_FILE}")

# === EXTRACT UNIQUE IDs ===
unique_ids = sorted(set(df["SWISS-PROT Accessions Interactor A"]) | set(df["SWISS-PROT Accessions Interactor B"]))
print(f"[INFO] Unique SWISS-PROT IDs: {len(unique_ids):,}")

# === RETRIEVE PROTEIN DATA FROM UniProt ===
def fetch_protein_data(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            js = response.json()
            name = js.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
            seq = js.get("sequence", {}).get("value")
            return name, seq
        elif response.status_code == 404:
            return None, None
        else:
            return None, None
    except Exception as e:
        print(f"[WARN] Failed {protein_id}: {e}")
        return None, None

protein_names = {}
protein_sequences = {}

for pid in tqdm(unique_ids, desc="Fetching from UniProt"):
    name, seq = fetch_protein_data(pid)
    if name and seq:
        protein_names[pid] = name
        protein_sequences[pid] = seq
    time.sleep(0.5)

protein_df = pd.DataFrame({
    "protein_id": list(protein_names.keys()),
    "protein_name": list(protein_names.values()),
    "protein_sequence": list(protein_sequences.values())
})
protein_df.to_csv(OUTPUT_DIR / "protein_data_raw.csv", index=False)
print(f"[INFO] Protein metadata saved ({len(protein_df):,} records).")

# === MERGE BACK INTO MAIN TABLE ===
id_to_name = dict(zip(protein_df["protein_id"], protein_df["protein_name"]))
id_to_seq = dict(zip(protein_df["protein_id"], protein_df["protein_sequence"]))

df["protein_name_a"] = df["SWISS-PROT Accessions Interactor A"].map(id_to_name)
df["protein_sequence_a"] = df["SWISS-PROT Accessions Interactor A"].map(id_to_seq)
df["protein_name_b"] = df["SWISS-PROT Accessions Interactor B"].map(id_to_name)
df["protein_sequence_b"] = df["SWISS-PROT Accessions Interactor B"].map(id_to_seq)

# === CLEAN AND DROP DUPLICATES ===
df.dropna(subset=["protein_sequence_a", "protein_sequence_b"], inplace=True)
df.drop_duplicates(subset=["protein_name_a", "protein_name_b", "protein_sequence_a", "protein_sequence_b"], inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv(FINAL_FILE, index=False)
print(f"[INFO] Final combined data saved to {FINAL_FILE} ({len(df):,} rows).")

# === CREATE ENTITY DATASETS ===
protein_data_a = df[["protein_name_a", "protein_sequence_a", "Qualifications"]].rename(columns={
    "protein_name_a": "protein_name",
    "protein_sequence_a": "protein_sequence",
    "Qualifications": "annotation",
})
protein_data_b = df[["protein_name_b", "protein_sequence_b", "Qualifications"]].rename(columns={
    "protein_name_b": "protein_name",
    "protein_sequence_b": "protein_sequence",
    "Qualifications": "annotation",
})
protein_data = pd.concat([protein_data_a, protein_data_b], ignore_index=True).drop_duplicates(subset=["protein_name", "protein_sequence"])
protein_data["type"] = "protein"
protein_data["subtype"] = None
protein_data["representation_type"] = "sequence"
protein_data.to_csv(PROTEIN_FILE, index=False)
print(f"[INFO] Protein dataset saved to {PROTEIN_FILE} ({len(protein_data):,} unique proteins).")

# === ANALYZE AMBIGUITIES ===
print("[INFO] Analyzing name/sequence ambiguities...")

# name to sequences
name_to_sequences = protein_data.groupby("protein_name")["protein_sequence"].unique()
name_multi = name_to_sequences[name_to_sequences.map(len) > 1]

# sequence to names
seq_to_names = protein_data.groupby("protein_sequence")["protein_name"].unique()
seq_multi = seq_to_names[seq_to_names.map(len) > 1]

# Save CSVs
name_multi.reset_index().to_csv(NAME_TO_SEQ, index=False)
seq_multi.reset_index().to_csv(SEQ_TO_NAME, index=False)

# Save TXT summary
with open(AMBIGUITY_TXT, "w", encoding="utf-8") as f:
    f.write(f"Total proteins: {len(protein_data):,}\n")
    f.write(f"Name to Sequences ambiguous cases: {len(name_multi):,}\n")
    f.write(f"Sequence to Names ambiguous cases: {len(seq_multi):,}\n")
    f.write("\nTop examples (Name to Sequences):\n")
    for name, seqs in name_multi.head(5).items():
        f.write(f"  {name}: {len(seqs)} variants\n")
    f.write("\nTop examples (Sequence to Names):\n")
    for seq, names in seq_multi.head(5).items():
        f.write(f"  seq[{seq[:15]}...]: {len(names)} variants\n")

print(f"[INFO] Ambiguity statistics saved to {AMBIGUITY_TXT}")
print(f"[INFO] name to seq CSV: {NAME_TO_SEQ}")
print(f"[INFO] seq to name CSV: {SEQ_TO_NAME}")

# === SUMMARY STATS ===
stats = {
    "initial_rows": initial_len,
    "filtered_rows": filtered_len,
    "unique_proteins": len(unique_ids),
    "retrieved_from_uniprot": len(protein_df),
    "final_rows": len(df),
    "name_to_sequences_ambiguous": len(name_multi),
    "sequence_to_names_ambiguous": len(seq_multi),
}
with open(STATS_FILE, "w", encoding="utf-8") as f:
    for k, v in stats.items():
        f.write(f"{k}: {v}\n")
print(f"[INFO] Statistics saved to {STATS_FILE}")
print("[DONE] Biogrid processing pipeline completed successfully.")