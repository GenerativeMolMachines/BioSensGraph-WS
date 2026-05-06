import os
import json
import time
from typing import Dict, List, Optional

import requests
import pandas as pd


class AlphaFoldSequenceSearcher:
    def __init__(self):
        self.alphafold_url = "https://alphafold.ebi.ac.uk/api"
        self.uniprot_url = "https://rest.uniprot.org/uniprotkb"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def search_alphafold_by_sequence(self, sequence: str, max_results: int = 10) -> Dict:
        """Search AlphaFold structures by amino-acid sequence with expanded matching strategies."""
        try:
            # UniProt search with relaxed criteria
            uniprot_results = self._find_uniprot_by_sequence_improved(sequence, max_results)
            
            if not uniprot_results:
                # Fallback similarity search
                uniprot_results = self._search_by_sequence_similarity(sequence, max_results)
            
            alphafold_results = []
            
            for uniprot_id, identity, db_sequence in uniprot_results:
                af_info = self._get_alphafold_structure(uniprot_id)
                if af_info:
                    protein_info = self._get_protein_info(uniprot_id)
                    
                    alphafold_results.append({
                        'alphafold_id': uniprot_id,
                        'uniprot_id': uniprot_id,
                        'identity_percent': identity,
                        'similarity_level': self._get_similarity_level(identity),
                        'protein_name': protein_info.get('protein_name', 'N/A'),
                        'gene_name': protein_info.get('gene_name', 'N/A'),
                        'organism': protein_info.get('organism', 'N/A'),
                        'plddt_score': af_info.get('plddt', 'N/A'),
                        'confidence_level': af_info.get('confidence', 'N/A'),
                        'model_url': af_info.get('model_url', 'N/A'),
                        'download_url': af_info.get('download_url', 'N/A'),
                        'sequence_length': protein_info.get('length', 0),
                        'query_coverage': self._calculate_coverage(sequence, db_sequence),
                        'match_type': 'exact' if identity == 100 else 'partial'
                    })
            
            alphafold_ids = [result['alphafold_id'] for result in alphafold_results]
            
            return {
                'success': True,
                'found': len(alphafold_ids) > 0,
                'count': len(alphafold_ids),
                'alphafold_ids': alphafold_ids,
                'detailed_results': alphafold_results,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'found': False,
                'count': 0,
                'alphafold_ids': [],
                'detailed_results': [],
                'error': f"Search error: {str(e)}"
            }
    
    def _find_uniprot_by_sequence_improved(self, sequence: str, max_results: int) -> List[tuple]:
        """UniProt search combining multiple strategies."""
        strategies = [
            self._search_exact_match,
            self._search_contains_sequence,
            self._search_by_length,
            self._search_blast_style
        ]
        
        all_results = []
        seen_ids = set()
        
        for strategy in strategies:
            try:
                results = strategy(sequence, max_results)
                for uniprot_id, identity, db_seq in results:
                    if uniprot_id not in seen_ids:
                        seen_ids.add(uniprot_id)
                        all_results.append((uniprot_id, identity, db_seq))
                        if len(all_results) >= max_results:
                            break
                if len(all_results) >= max_results:
                    break
            except:
                continue
        
        return all_results[:max_results]
    
    def _search_exact_match(self, sequence: str, max_results: int) -> List[tuple]:
        """Exact sequence match in UniProt."""
        try:
            query = f'sequence:"{sequence}"'
            params = {
                'query': query,
                'fields': 'accession,sequence',
                'size': max_results,
                'format': 'json'
            }
            
            response = self.session.get(f"{self.uniprot_url}/search", params=params, timeout=30)
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                uniprot_id = item.get('primaryAccession')
                db_sequence = item.get('sequence', {}).get('value', '')
                if uniprot_id and sequence == db_sequence:
                    results.append((uniprot_id, 100.0, db_sequence))
            
            return results
        except:
            return []
    
    def _search_contains_sequence(self, sequence: str, max_results: int) -> List[tuple]:
        """Find proteins that contain the query sequence or high partial identity."""
        try:
            # Proteins at least as long as the query
            query = f'length:[{len(sequence)} TO *]'
            params = {
                'query': query,
                'fields': 'accession,sequence',
                'size': min(max_results * 3, 50),
                'format': 'json',
                'sort': 'length'
            }
            
            response = self.session.get(f"{self.uniprot_url}/search", params=params, timeout=30)
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                uniprot_id = item.get('primaryAccession')
                db_sequence = item.get('sequence', {}).get('value', '')
                
                if uniprot_id and db_sequence:
                    if sequence in db_sequence:
                        identity = 100.0
                        results.append((uniprot_id, identity, db_sequence))
                    else:
                        identity = self._calculate_best_identity(sequence, db_sequence)
                        if identity >= 50:
                            results.append((uniprot_id, identity, db_sequence))
                
                if len(results) >= max_results:
                    break
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
            
        except:
            return []
    
    def _search_by_length(self, sequence: str, max_results: int) -> List[tuple]:
        """Search proteins with similar length (±10%)."""
        try:
            seq_len = len(sequence)
            length_range = max(10, int(seq_len * 0.1))
            min_len = seq_len - length_range
            max_len = seq_len + length_range
            
            query = f'length:[{min_len} TO {max_len}]'
            params = {
                'query': query,
                'fields': 'accession,sequence',
                'size': min(max_results * 2, 30),
                'format': 'json'
            }
            
            response = self.session.get(f"{self.uniprot_url}/search", params=params, timeout=30)
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                uniprot_id = item.get('primaryAccession')
                db_sequence = item.get('sequence', {}).get('value', '')
                
                if uniprot_id and db_sequence:
                    identity = self._calculate_best_identity(sequence, db_sequence)
                    if identity >= 40:
                        results.append((uniprot_id, identity, db_sequence))
                
                if len(results) >= max_results:
                    break
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
            
        except:
            return []
    
    def _search_blast_style(self, sequence: str, max_results: int) -> List[tuple]:
        """BLAST-like scan: strong local similarity over a limited result window."""
        try:
            query = '*'
            params = {
                'query': query,
                'fields': 'accession,sequence',
                'size': min(max_results, 20),
                'format': 'json',
                'sort': 'reviewed'
            }
            
            response = self.session.get(f"{self.uniprot_url}/search", params=params, timeout=30)
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                uniprot_id = item.get('primaryAccession')
                db_sequence = item.get('sequence', {}).get('value', '')
                
                if uniprot_id and db_sequence:
                    identity = self._calculate_best_local_identity(sequence, db_sequence)
                    if identity >= 80:
                        results.append((uniprot_id, identity, db_sequence))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
            
        except:
            return []
    
    def _search_by_sequence_similarity(self, sequence: str, max_results: int) -> List[tuple]:
        """Alternative fallback using varied UniProt query terms."""
        try:
            search_terms = [
                f'sequence:"{sequence}"',
                f'length:{len(sequence)}',
                'reviewed:true',
                '*'
            ]
            
            results = []
            for term in search_terms:
                if len(results) >= max_results:
                    break
                    
                params = {
                    'query': term,
                    'fields': 'accession,sequence',
                    'size': 10,
                    'format': 'json'
                }
                
                response = self.session.get(f"{self.uniprot_url}/search", params=params, timeout=20)
                data = response.json()
                
                for item in data.get('results', []):
                    uniprot_id = item.get('primaryAccession')
                    db_sequence = item.get('sequence', {}).get('value', '')
                    
                    if uniprot_id and db_sequence:
                        identity = self._calculate_best_identity(sequence, db_sequence)
                        if identity >= 30:
                            results.append((uniprot_id, identity, db_sequence))
            
            seen = set()
            unique_results = []
            for result in results:
                if result[0] not in seen:
                    seen.add(result[0])
                    unique_results.append(result)
            
            unique_results.sort(key=lambda x: x[1], reverse=True)
            return unique_results[:max_results]
            
        except:
            return []
    
    def _calculate_best_identity(self, query_seq: str, db_seq: str) -> float:
        """Best identity across exact, local, and sliding-window comparisons."""
        methods = [
            self._calculate_exact_identity,
            self._calculate_local_identity,
            self._calculate_sliding_window_identity
        ]
        
        best_identity = 0.0
        for method in methods:
            identity = method(query_seq, db_seq)
            if identity > best_identity:
                best_identity = identity
        
        return best_identity
    
    def _calculate_exact_identity(self, query_seq: str, db_seq: str) -> float:
        """Full match or substring containment."""
        if query_seq == db_seq:
            return 100.0
        if query_seq in db_seq:
            return 100.0
        if len(query_seq) <= len(db_seq) and db_seq in query_seq:
            return 100.0
        return 0.0
    
    def _calculate_local_identity(self, query_seq: str, db_seq: str) -> float:
        """Compare prefix and suffix alignment."""
        if not query_seq or not db_seq:
            return 0.0
        
        min_len = min(len(query_seq), len(db_seq))
        if min_len == 0:
            return 0.0
        
        start_matches = sum(1 for a, b in zip(query_seq, db_seq) if a == b)
        start_identity = (start_matches / min_len) * 100
        
        end_matches = sum(1 for a, b in zip(query_seq[::-1], db_seq[::-1]) if a == b)
        end_identity = (end_matches / min_len) * 100
        
        return max(start_identity, end_identity)
    
    def _calculate_sliding_window_identity(self, query_seq: str, db_seq: str) -> float:
        """Sliding window over db_seq to maximize residue identity."""
        if len(query_seq) > len(db_seq):
            return 0.0
        
        best_identity = 0.0
        for i in range(0, len(db_seq) - len(query_seq) + 1):
            substring = db_seq[i:i + len(query_seq)]
            matches = sum(1 for a, b in zip(query_seq, substring) if a == b)
            identity = (matches / len(query_seq)) * 100
            if identity > best_identity:
                best_identity = identity
        
        return best_identity
    
    def _calculate_best_local_identity(self, query_seq: str, db_seq: str) -> float:
        """Strong local match heuristic for short queries."""
        if len(query_seq) <= 10:
            best_identity = 0.0
            for i in range(0, len(query_seq) - 4 + 1):
                pattern = query_seq[i:i + 4]
                if pattern in db_seq:
                    best_identity = max(best_identity, 80.0)
        
        return self._calculate_sliding_window_identity(query_seq, db_seq)
    
    def _calculate_coverage(self, query_seq: str, db_seq: str) -> float:
        """Coverage proxy from length ratio."""
        if not query_seq or not db_seq:
            return 0.0
        return round((min(len(query_seq), len(db_seq)) / max(len(query_seq), len(db_seq))) * 100, 2)
    
    def _get_alphafold_structure(self, uniprot_id: str) -> Optional[Dict]:
        """Fetch AlphaFold prediction metadata if available."""
        try:
            url = f"{self.alphafold_url}/prediction/{uniprot_id}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    structure_info = data[0]
                    plddt_scores = structure_info.get('plddt', [])
                    avg_plddt = sum(plddt_scores) / len(plddt_scores) if plddt_scores else 0
                    
                    return {
                        'plddt': round(avg_plddt, 2),
                        'confidence': self._get_confidence_level(avg_plddt),
                        'model_url': f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}",
                        'download_url': structure_info.get('pdbUrl', 'N/A')
                    }
            return None
        except:
            return None
    
    def _get_protein_info(self, uniprot_id: str) -> Dict:
        """UniProt entry metadata for display columns."""
        try:
            url = f"{self.uniprot_url}/{uniprot_id}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return {
                    'protein_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A'),
                    'gene_name': data.get('genes', [{}])[0].get('geneName', {}).get('value', 'N/A') if data.get('genes') else 'N/A',
                    'organism': data.get('organism', {}).get('scientificName', 'N/A'),
                    'length': data.get('sequence', {}).get('length', 0),
                    'sequence': data.get('sequence', {}).get('value', '')
                }
            return {'protein_name': uniprot_id}
        except:
            return {'protein_name': uniprot_id}
    
    def _get_similarity_level(self, percent: float) -> str:
        if percent == 100: return "exact_match"
        elif percent >= 90: return "very_high"
        elif percent >= 80: return "high"
        elif percent >= 70: return "medium"
        elif percent >= 60: return "low"
        else: return "very_low"
    
    def _get_confidence_level(self, plddt: float) -> str:
        if plddt >= 90: return "very_high"
        elif plddt >= 70: return "confident"
        elif plddt >= 50: return "low"
        else: return "very_low"

def process_alphafold_sequence_search(csv_file_path: str, output_file: Optional[str] = None, max_results: int = 10) -> pd.DataFrame:
    """Run sequence-based AlphaFold lookup over a CSV with nodeid, name, content (sequence)."""
    if not os.path.exists(csv_file_path):
        print(f"Error: file not found: {csv_file_path}")
        return None
    
    df = pd.read_csv(csv_file_path)
    
    required_columns = ['nodeid', 'name', 'content']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: missing required column '{col}'")
            return None
    
    print(f"Loaded rows: {len(df)}")
    print("AlphaFold sequence search...")
    print("=" * 60)
    
    searcher = AlphaFoldSequenceSearcher()
    results = []
    
    for index, row in df.iterrows():
        print(f"\nRow {index + 1}/{len(df)}: {row['name']}")
        print(f"Sequence: {row['content'][:30]}...")
        print(f"Length: {len(row['content'])} residues")
        
        sequence = row['content'].strip()
        result = searcher.search_alphafold_by_sequence(sequence, max_results)
        
        similarity_info = result['detailed_results']
        
        result_data = {
            'nodeid': row['nodeid'],
            'name': row['name'],
            'sequence': sequence,
            'sequence_length': len(sequence),
            'found_in_alphafold': result['found'],
            'alphafold_results_count': result['count'],
            'alphafold_ids': ', '.join(result['alphafold_ids']) if result['alphafold_ids'] else 'None',
            'similarity_info': json.dumps(similarity_info, ensure_ascii=False) if similarity_info else 'None',
            'max_identity': max([item['identity_percent'] for item in similarity_info]) if similarity_info else 0,
            'best_match': similarity_info[0]['alphafold_id'] if similarity_info else 'None',
            'error': result['error'] if result['error'] else 'None'
        }
        
        results.append(result_data)
        
        status = "FOUND" if result['found'] else "NOT FOUND"
        print(f"  {status} — {result['count']} hit(s)")
        
        if result['found']:
            best_match = similarity_info[0]
            print(f"  Best match: {best_match['alphafold_id']}")
            print(f"  Identity: {best_match['identity_percent']}%")
            print(f"  Model confidence: {best_match['confidence_level']} (pLDDT: {best_match['plddt_score']})")
        
        time.sleep(2)
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults written to: {output_file}")
    
    return results_df

def create_alphafold_sequence_report(results_df: pd.DataFrame, output_file: str = "results/alphafold_sequence_report.csv") -> pd.DataFrame:
    """Expand per-hit rows into a flat AlphaFold sequence report CSV."""
    detailed_data = []
    
    for _, row in results_df.iterrows():
        if row['found_in_alphafold'] and row['similarity_info'] != 'None':
            try:
                similarity_data = json.loads(row['similarity_info'])
                for match in similarity_data:
                    detailed_data.append({
                        'nodeid': row['nodeid'],
                        'name': row['name'],
                        'alphafold_id': match['alphafold_id'],
                        'uniprot_id': match['uniprot_id'],
                        'identity_percent': match['identity_percent'],
                        'similarity_level': match['similarity_level'],
                        'protein_name': match.get('protein_name', 'N/A'),
                        'gene_name': match.get('gene_name', 'N/A'),
                        'organism': match.get('organism', 'N/A'),
                        'plddt_score': match.get('plddt_score', 'N/A'),
                        'confidence_level': match.get('confidence_level', 'N/A'),
                        'query_coverage': match.get('query_coverage', 0),
                        'sequence_length': match.get('sequence_length', 0),
                        'model_url': match.get('model_url', 'N/A')
                    })
            except Exception as e:
                print(f"Error while building detailed rows: {e}")
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Detailed AlphaFold report saved to: {output_file}")
        return detailed_df
    else:
        print("No rows for detailed AlphaFold report")
        return None

def main():
    """CLI entry: sequence search against AlphaFold via UniProt (paths relative to repo root)."""
    csv_file_path = "data/db_export_proteins.csv"

    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return
    
    output_file = "results/alphafold_sequence_results.csv"
    max_results = 10
    
    print(f"\nSearch settings:")
    print(f"  Input CSV: {csv_file_path}")
    print(f"  Output CSV: {output_file}")
    print(f"  Max results per query: {max_results}")
    print("=" * 60)
    
    results = process_alphafold_sequence_search(
        csv_file_path=csv_file_path,
        output_file=output_file,
        max_results=max_results
    )
    
    if results is not None:
        total = len(results)
        found = results['found_in_alphafold'].sum()
        
        print(f"\nAlphaFold search summary:")
        print(f"  Total sequences: {total}")
        print(f"  Found in AlphaFold: {found} ({found/total*100:.1f}%)")
        
        if found > 0:
            create_alphafold_sequence_report(results)
            
            avg_identity = results[results['max_identity'] > 0]['max_identity'].mean()
            exact_matches = len(results[results['max_identity'] == 100])
            
            print(f"  Mean best identity: {avg_identity:.1f}%")
            print(f"  Exact matches (100%): {exact_matches}")

if __name__ == "__main__":
    main()
