import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import csv
import re
from collections import Counter
from pathlib import Path


# ====== CONFIG ===========

SDF_FILE = "../../data/BindingDB_All_2D.sdf"
EXTRACTED_CSV = "../../data/binding_db_selected_properties.csv"
CLEAN_CSV = "../../data/binding_db_clean.csv"
REPORT_FILE = "../../data/invalid_sequences_report.txt"
OUT_DIR = Path("../../data/processed_binding_db")
OUT_DIR.mkdir(exist_ok=True, parents=True)

REQUIRED_PROPERTIES = [
    "BindingDB Ligand Name",
    "Target Source Organism According to Curator or DataSource",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "pH",
    "Temp C",
    "PubChem CID of Ligand",
    "BindingDB Target Chain Sequence",
    "UniProt (SwissProt) Recommended Name of Target Chain",
    "UniProt (SwissProt) Primary ID of Target Chain",
]

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
SYMBOLS_TO_REMOVE = {'<', '>', ',', ' '}


# ====== FUNCTIONS ========


def extract_selected_properties(sdf_file: str, output_csv: str, required_properties: list[str]):
    """Extract selected properties from BindingDB SDF file and save to CSV."""
    suppl = Chem.SDMolSupplier(sdf_file)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(required_properties + ["smiles"])

        for mol in tqdm(suppl, desc="Extracting selected properties", unit="molecule"):
            if mol is None:
                continue
            row = [mol.GetProp(p) if mol.HasProp(p) else None for p in required_properties]
            row.append(Chem.MolToSmiles(mol) if mol is not None else None)
            writer.writerow(row)

    print(f"[INFO] Extracted properties saved to {output_csv}")


def convert_to_float(value, symbols_to_remove=None, error_values=None):
    """Clean and convert value to float; handle exponential and malformed strings."""
    if pd.isna(value) or value in ['', ' ']:
        return None
    try:
        cleaned = ''.join(c for c in str(value) if c not in symbols_to_remove).strip()
        if 'e' in cleaned.lower():
            base, exponent = cleaned.lower().split('e')
            exponent = exponent.replace('+', '')
            return float(base) * (10 ** int(exponent))
        return float(cleaned)
    except (ValueError, TypeError):
        if error_values is not None:
            error_values.add(value)
        return None


def validate_protein(seq: str) -> tuple[str | None, str | None]:
    """Validate protein sequence (AA symbols, case, format)."""
    if not isinstance(seq, str) or not seq.strip():
        return None, "Empty or non-string"
    seq = re.sub(r"\s+|\d+", "", seq)
    if any(c.islower() for c in seq):
        return None, "Lowercase letters detected"
    invalid_chars = {c for c in seq if c.upper() not in VALID_AA}
    if invalid_chars:
        return None, f"Invalid amino acids: {', '.join(sorted(invalid_chars))}"
    return seq, None


def validate_smiles(smiles: str) -> tuple[str | None, str | None]:
    """Validate SMILES string using RDKit."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None, "Empty or non-string"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    return smiles, None


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES representation or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def collect_ambiguities(df, key_col, val_col, out_path):
    """Aggregate one-to-many relationships and save to CSV."""
    agg = (
        df.groupby(key_col)[val_col]
        .agg(lambda s: sorted(set(s.dropna())))
        .reset_index()
    )
    agg["count"] = agg[val_col].apply(len)
    agg = agg.sort_values("count", ascending=False)
    agg.to_csv(out_path, index=False)
    return agg



# ====== PIPELINE =========


if __name__ == "__main__":
    # Extract properties from SDF
    extract_selected_properties(SDF_FILE, EXTRACTED_CSV, REQUIRED_PROPERTIES)

    # Initial cleaning
    df = pd.read_csv(EXTRACTED_CSV)
    print(f"[INFO] Imported {len(df):,} rows")

    # Drop incomplete rows
    df = df.dropna(subset=[
        "UniProt (SwissProt) Recommended Name of Target Chain",
        "BindingDB Target Chain Sequence",
        "BindingDB Ligand Name",
        "smiles",
    ])
    df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

    df = df.drop_duplicates(subset=[
        "UniProt (SwissProt) Recommended Name of Target Chain",
        "BindingDB Target Chain Sequence",
        "BindingDB Ligand Name",
        "smiles",
    ]).reset_index(drop=True)

    print(f"[INFO] After deduplication: {len(df):,} rows")

    # Numeric cleanup
    error_values = set()
    df["Kd (nM)"] = df.apply(
        lambda r: r["Ki (nM)"] if pd.isna(r["Kd (nM)"]) else r["Kd (nM)"], axis=1
    )
    df["Kd (nM)"] = df["Kd (nM)"].apply(lambda x: convert_to_float(x, SYMBOLS_TO_REMOVE, error_values))
    print(f"[INFO] Invalid numeric values: {error_values}")

    # Validate sequences and SMILES
    df[["protein_sequence", "protein_issue"]] = df["BindingDB Target Chain Sequence"].apply(
        lambda s: pd.Series(validate_protein(s))
    )
    df[["small_molecule_smiles", "smiles_issue"]] = df["smiles"].apply(
        lambda s: pd.Series(validate_smiles(s))
    )

    before = len(df)
    df_valid = df.dropna(subset=["protein_sequence", "small_molecule_smiles"]).reset_index(drop=True)
    after = len(df_valid)
    print(f"[INFO] Valid records: {after:,} / {before:,}")

    df_valid.to_csv(CLEAN_CSV, index=False)
    print(f"[INFO] Clean dataset saved to {CLEAN_CSV}")

    # Report invalid records
    invalid_df = df[(df["protein_sequence"].isna()) | (df["small_molecule_smiles"].isna())]
    issue_counts = Counter(invalid_df["protein_issue"].dropna()) + Counter(invalid_df["smiles_issue"].dropna())

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== INVALID RECORDS REPORT ===\n\n")
        f.write("Summary of issues:\n")
        for issue, count in issue_counts.items():
            f.write(f"  {issue}: {count}\n")
        f.write("\n\nExamples:\n")
        for issue_type in issue_counts.keys():
            subset = invalid_df[
                (invalid_df["protein_issue"] == issue_type) | (invalid_df["smiles_issue"] == issue_type)
            ]
            f.write(f"\n--- {issue_type} ---\n")
            for _, row in subset.head(3).iterrows():
                f.write(f"Protein: {row.get('BindingDB Target Chain Sequence', '')}\n")
                f.write(f"SMILES: {row.get('smiles', '')}\n\n")
    print(f"[INFO] Invalid sequence report saved to {REPORT_FILE}")

    # Canonicalize SMILES and finalize
    print("[INFO] Canonicalizing SMILES...")
    tqdm.pandas()
    df = pd.read_csv(CLEAN_CSV)
    df = df.rename(columns={
        'BindingDB Ligand Name': 'molecule_name',
        'UniProt (SwissProt) Recommended Name of Target Chain': 'protein_name'
    })
    df = df.drop(columns=['smiles', 'protein_issue', 'smiles_issue', 'BindingDB Target Chain Sequence'])
    df["molecule_smiles"] = df["small_molecule_smiles"].progress_apply(canonicalize_smiles)
    df = df.drop(columns=['small_molecule_smiles']).dropna(subset=["molecule_smiles"]).reset_index(drop=True)

    # Remove duplicate interactions
    cols_unique = ["protein_name", "protein_sequence", "molecule_name", "molecule_smiles"]
    df_unique = df.drop_duplicates(subset=cols_unique).reset_index(drop=True)
    print(f"[INFO] Unique interactions: {len(df_unique):,}")

    # Save unique interactions
    main_cols = ["protein_name", "protein_sequence", "molecule_name", "molecule_smiles"]
    other_cols = [c for c in df_unique.columns if c not in main_cols]
    df_unique = df_unique[main_cols + other_cols]
    interactions_path = OUT_DIR / "binding_db_interactions_unique.csv"
    df_unique.to_csv(interactions_path, index=False)
    print(f"[INFO] Unique interactions saved to {interactions_path}")

    # Save entities (proteins & molecules)
    proteins = df_unique[["protein_name", "protein_sequence"]].drop_duplicates().reset_index(drop=True)
    molecules = df_unique[["molecule_name", "molecule_smiles"]].drop_duplicates().reset_index(drop=True)
    proteins.to_csv(OUT_DIR / "binding_db_proteins.csv", index=False)
    molecules.to_csv(OUT_DIR / "binding_db_molecules.csv", index=False)
    print(f"[INFO] Saved entities:\n - Proteins: {len(proteins):,}\n - Molecules: {len(molecules):,}")

    # Ambiguity statistics
    print("[INFO] Calculating ambiguity statistics...")
    protein_seq_to_names = collect_ambiguities(df_unique, "protein_sequence", "protein_name", OUT_DIR / "protein_seq_to_names.csv")
    protein_name_to_seq = collect_ambiguities(df_unique, "protein_name", "protein_sequence", OUT_DIR / "protein_name_to_seq.csv")
    molecule_smiles_to_names = collect_ambiguities(df_unique, "molecule_smiles", "molecule_name", OUT_DIR / "molecule_smiles_to_names.csv")
    molecule_name_to_smiles = collect_ambiguities(df_unique, "molecule_name", "molecule_smiles", OUT_DIR / "molecule_name_to_smiles.csv")

    summary = {
        "total_clean_interactions": len(df),
        "unique_interactions": len(df_unique),
        "unique_proteins": len(proteins),
        "unique_molecules": len(molecules),
        "protein_seq_multi_names": (protein_seq_to_names["count"] > 1).sum(),
        "protein_name_multi_seq": (protein_name_to_seq["count"] > 1).sum(),
        "smiles_multi_names": (molecule_smiles_to_names["count"] > 1).sum(),
        "name_multi_smiles": (molecule_name_to_smiles["count"] > 1).sum(),
    }

    summary_path = OUT_DIR / "ambiguity_stats.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\nSUMMARY")
    for k, v in summary.items():
        print(f" - {k}: {v}")
    print(f"[INFO] Detailed stats saved to {summary_path}")