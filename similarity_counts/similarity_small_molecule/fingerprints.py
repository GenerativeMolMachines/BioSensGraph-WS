from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm

def generate_fingerprints(df, radius, n_bits):
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, bit_counts, names, sequences = [], [], [], []

    for name, smi in tqdm(zip(df["id"], df["content"]),
                          total=len(df), desc="Fingerprint gen"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Invalid SMILES: {name} ({smi})")
            continue
        fp = gen.GetFingerprint(mol)
        fps.append(fp)
        bit_counts.append(fp.GetNumOnBits())
        names.append(name)
        sequences.append(smi)

    return fps, bit_counts, names, sequences

