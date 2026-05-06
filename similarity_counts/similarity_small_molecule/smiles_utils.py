from rdkit import Chem
import csv
from pathlib import Path

def smi_canon_and_log(smi, name, bad_writer=None):
 """Kanonizirovat SMILES, logirovat nevalidnye"""
    if not isinstance(smi, str) or not smi.strip():
        if bad_writer:
            bad_writer.writerow([name, smi])
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if bad_writer:
                bad_writer.writerow([name, smi])
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        if bad_writer:
            bad_writer.writerow([name, smi])
        return None
