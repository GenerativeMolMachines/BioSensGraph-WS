from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
in_path = ROOT / "data" / "data_small_molecule" / "smiles_df.csv"
out_path = ROOT / "data" / "data_small_molecule" / "similarity_small_molecule.csv"

threshold = 0.80
radius = 2
n_bits = 2048
n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
