import os
import time
import logging
from urllib.parse import quote

import requests
import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(filename)s:%(funcName)s:%(message)s'
)

logger = logging.getLogger(__name__)

# Paths below are relative to the repository root (run scripts with cwd at repo root).

if not os.path.exists('tmp'):
    os.makedirs('tmp', exist_ok=True)


class MoleculeCrossRefParser:
    def __init__(self, delay=0.5):
        """
        Initialize the MoleculeCrossRefParser.

        Args:
            delay (float): Delay between API calls to avoid overloading servers.
        """
        self.delay = delay

    @staticmethod
    def _prepare_smiles(raw_smiles: str) -> str:
        return quote(raw_smiles, safe='')


    def _get_chembl_id(self, smiles=None, name=None):
        base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
        if smiles:
            smiles = self._prepare_smiles(smiles)
            query = f"?molecule_structures__canonical_smiles__flexmatch={smiles}"
        elif name:
            query = f"?molecule_synonyms__icontains={name}"
        else:
            return None

        try:
            r = requests.get(base_url + query, headers={"Accept": "application/json"}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data["page_meta"]["total_count"] > 0:
                    return data["molecules"][0]["molecule_chembl_id"]
        except Exception as e:
            logger.error(f"Cannot get CHEMBL_ID for {name} {smiles} due to the error: {e}")
        return None

    def _get_pubchem_cid(self, smiles=None, name=None):
        if smiles:
            smiles = self._prepare_smiles(smiles)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/cids/JSON"
        elif name:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
        else:
            return None

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                    return data["IdentifierList"]["CID"][0]
        except Exception as e:
            logger.error(f"Cannot get PUBCHEM CID for {name} {smiles} due to the error: {e}")
        return None


    def get_molecule_ids(self, smiles=None, name=None):
        """Return molecule identifiers (CHEMBL_ID, PUBCHEM_CID)."""
        key = smiles or name
        if not key:
            return {"CHEMBL_ID": None, "PUBCHEM_CID": None}

        chembl_id = self._get_chembl_id(smiles, name)
        pubchem_cid = self._get_pubchem_cid(smiles, name)
        result = {
            "CHEMBL_ID": chembl_id,
            "PUBCHEM_CID": pubchem_cid,
        }

        # Polite delay to avoid API throttling
        time.sleep(self.delay)
        return result
    
    def get_chembl_name(self, chembl_id: str):
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                return {
                    "chembl_preferred_name": data.get("pref_name"),
                    "chembl_synonyms": [s["synonyms"] for s in data.get("molecule_synonyms", [])]
                }
        except Exception as e:
            logger.error(f"Cannot get CHEMBL name for {chembl_id} due to the error: {e}")
        return {
                "chembl_preferred_name": None,
                "chembl_synonyms": None,
            }

    def get_pubchem_name(self, cid):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,Title/JSON"
        try:
            r = requests.get(url)
            if r.status_code == 200:
                props = r.json()["PropertyTable"]["Properties"][0]
                return {
                    "pubchem_preferred_name": props.get("Title"),
                    "iupac_name": props.get("IUPACName")
                }
        except Exception as e:
            logger.error(f"Cannot get PUBCHEM name for {cid} due to the error: {e}")
        return {
            "pubchem_preferred_name": None,
            "iupac_name": None
        }


    def process_csv(self, input_csv_path: str, output_csv: str):
        df = pd.read_csv(input_csv_path)

        results = []
        for row in tqdm(df.to_dict("records")[:100]):
            smiles = row["smiles"]

            ids = self.get_molecule_ids(smiles)
            result_row = row
            result_row.update(ids)

            if ids['PUBCHEM_CID'] is not None and ids['PUBCHEM_CID'] != 0.0:
                pubchem_names = self.get_pubchem_name(ids['PUBCHEM_CID'])
                result_row.update(pubchem_names)
            
            if ids['CHEMBL_ID'] is not None:
                chembl_names = self.get_chembl_name(ids['CHEMBL_ID'])
                result_row.update(chembl_names)

            results.append(result_row)

        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    parser = MoleculeCrossRefParser()
    parser.process_csv(
        'data/small_molecules_full.csv',
        'data/crossref/small_molecules/pubchem_cheml.csv',
    )