import requests
import pandas as pd
import time
import json
import os
from typing import Dict, List, Optional

class AlphaFoldSearcher:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org"
        self.search_url = "https://rest.uniprot.org/uniprotkb/search"
        self.alphafold_url = "https://alphafold.ebi.ac.uk/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def search_alphafold_by_name(self, name: str, max_results: int = 10) -> Dict:
        """Resolve protein name to UniProt accessions and check AlphaFold coverage."""
        try:
            uniprot_results = self._search_uniprot_by_name(name, max_results)
            
            if not uniprot_results:
                return {
                    'success': True,
                    'found': False,
                    'count': 0,
                    'alphafold_ids': [],
                    'detailed_results': [],
                    'error': None
                }
            
            alphafold_ids = []
            detailed_results = []
            
            for uniprot_result in uniprot_results:
                uniprot_id = uniprot_result['uniprot_id']
                alphafold_id = self._get_alphafold_structure(uniprot_id)
                
                if alphafold_id:
                    alphafold_ids.append(alphafold_id)
                    
                    af_info = self._get_alphafold_info(alphafold_id)
                    
                    detailed_results.append({
                        'alphafold_id': alphafold_id,
                        'uniprot_id': uniprot_id,
                        'relevance_score': uniprot_result['relevance_score'],
                        'relevance_level': uniprot_result['relevance_level'],
                        'protein_name': uniprot_result['protein_name'],
                        'gene_name': uniprot_result['gene_name'],
                        'organism': uniprot_result['organism'],
                        'plddt_score': af_info.get('plddt', 'N/A'),
                        'confidence_level': af_info.get('confidence', 'N/A'),
                        'model_url': af_info.get('model_url', 'N/A'),
                        'sequence_length': uniprot_result.get('length', 0)
                    })
            
            return {
                'success': True,
                'found': len(alphafold_ids) > 0,
                'count': len(alphafold_ids),
                'alphafold_ids': alphafold_ids,
                'detailed_results': detailed_results,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'found': False,
                'count': 0,
                'alphafold_ids': [],
                'detailed_results': [],
                'error': f"AlphaFold search error: {str(e)}"
            }
    
    def _search_uniprot_by_name(self, name: str, max_results: int) -> List[Dict]:
        """UniProt text search by protein / gene name (internal)."""
        try:
            query = f'({name}) AND reviewed:true'
            params = {
                'query': query,
                'fields': 'accession,protein_name,gene_names,organism_name,sequence,length',
                'size': max_results,
                'format': 'json'
            }
            
            response = self.session.get(self.search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            uniprot_results = []
            
            for result in results:
                uniprot_id = result.get('primaryAccession')
                if uniprot_id:
                    protein_name = result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
                    gene_name = result.get('genes', [{}])[0].get('geneName', {}).get('value', 'N/A') if result.get('genes') else 'N/A'
                    
                    relevance_score = self._calculate_simple_relevance(name, protein_name, gene_name)
                    
                    if relevance_score >= 30:
                        uniprot_results.append({
                            'uniprot_id': uniprot_id,
                            'relevance_score': relevance_score,
                            'relevance_level': self._get_relevance_level(relevance_score),
                            'protein_name': protein_name,
                            'gene_name': gene_name,
                            'organism': result.get('organism', {}).get('scientificName', 'N/A'),
                            'length': result.get('sequence', {}).get('length', 0)
                        })
            
            uniprot_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return uniprot_results[:max_results]
            
        except Exception as e:
            print(f"UniProt search error: {e}")
            return []
    
    def _calculate_simple_relevance(self, query_name: str, protein_name: str, gene_name: str) -> float:
        """Heuristic relevance score between query and UniProt names."""
        query_lower = query_name.lower()
        protein_lower = protein_name.lower()
        gene_lower = gene_name.lower()
        
        if query_lower == protein_lower or query_lower == gene_lower:
            return 100.0
        
        if query_lower in protein_lower or query_lower in gene_lower:
            return 90.0
        
        query_words = query_lower.split()
        protein_words = protein_lower.split()
        gene_words = gene_lower.split()
        
        protein_matches = sum(1 for word in query_words if word in protein_lower)
        gene_matches = sum(1 for word in query_words if word in gene_lower)
        total_matches = protein_matches + gene_matches
        
        if total_matches > 0:
            return min(80.0, total_matches * 20.0)
        
        partial_matches = 0
        for q_word in query_words:
            for p_word in protein_words:
                if q_word in p_word or p_word in q_word:
                    partial_matches += 1
            for g_word in gene_words:
                if q_word in g_word or g_word in q_word:
                    partial_matches += 1
        
        if partial_matches > 0:
            return min(60.0, partial_matches * 15.0)
        
        return 30.0
    
    def _get_alphafold_structure(self, uniprot_id: str) -> Optional[str]:
        """Return UniProt ID if AlphaFold has a model; otherwise None."""
        try:
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return uniprot_id
            
            return None
            
        except Exception as e:
            print(f"AlphaFold check failed for {uniprot_id}: {e}")
            return None
    
    def _get_alphafold_info(self, alphafold_id: str) -> Dict:
        """pLDDT and URLs for one AlphaFold prediction."""
        try:
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{alphafold_id}"
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
                        'model_url': f"https://alphafold.ebi.ac.uk/entry/{alphafold_id}",
                        'download_url': structure_info.get('pdbUrl', 'N/A'),
                        'version': structure_info.get('version', 'N/A')
                    }
            
            return {}
            
        except:
            return {}
    
    def _get_relevance_level(self, score: float) -> str:
        """Bucket string for relevance score."""
        if score == 100:
            return "exact_match"
        elif score >= 90:
            return "very_high"
        elif score >= 80:
            return "high"
        elif score >= 70:
            return "medium"
        elif score >= 60:
            return "low"
        else:
            return "very_low"
    
    def _get_confidence_level(self, plddt: float) -> str:
        """Bucket string from mean pLDDT."""
        if plddt >= 90:
            return "very_high"
        elif plddt >= 70:
            return "confident"
        elif plddt >= 50:
            return "low"
        else:
            return "very_low"

def process_alphafold_search(csv_file_path: str, output_file: Optional[str] = None, max_results: int = 10) -> pd.DataFrame:
    """Batch AlphaFold name search over CSV columns nodeid, name, content."""
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
    print("AlphaFold lookup...")
    print("=" * 60)
    
    searcher = AlphaFoldSearcher()
    results = []
    
    for index, row in df.iterrows():
        print(f"\nRow {index + 1}/{len(df)}: {row['name']}")
        print(f"Query name: {row['name']}")
        
        protein_name = row['name'].strip()
        result = searcher.search_alphafold_by_name(protein_name, max_results)
        
        similarity_info = result['detailed_results']
        
        result_data = {
            'nodeid': row['nodeid'],
            'name': row['name'],
            'sequence': row['content'],
            'sequence_length': len(row['content']),
            'found_in_alphafold': result['found'],
            'alphafold_results_count': result['count'],
            'alphafold_ids': ', '.join(result['alphafold_ids']) if result['alphafold_ids'] else 'None',
            'similarity_info': json.dumps(similarity_info, ensure_ascii=False) if similarity_info else 'None',
            'max_relevance': max([item['relevance_score'] for item in similarity_info]) if similarity_info else 0,
            'best_match': similarity_info[0]['alphafold_id'] if similarity_info else 'None',
            'error': result['error'] if result['error'] else 'None'
        }
        
        results.append(result_data)
        
        status = "FOUND" if result['found'] else "NOT FOUND"
        print(f"  {status} — {result['count']} hit(s)")
        
        if result['found']:
            best_match = similarity_info[0]
            print(f"  Best match: {best_match['alphafold_id']}")
            print(f"  Relevance: {best_match['relevance_score']}%")
            if best_match.get('plddt_score') != 'N/A':
                print(f"  Model confidence: {best_match['confidence_level']} (pLDDT: {best_match['plddt_score']})")
        
        time.sleep(1)
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults written to: {output_file}")
    
    return results_df

def create_alphafold_detailed_report(results_df: pd.DataFrame, output_file: str = "results/alphafold_detailed_report.csv") -> pd.DataFrame:
    """Flatten JSON similarity column into one row per AlphaFold hit."""
    detailed_data = []
    
    for _, row in results_df.iterrows():
        if row['found_in_alphafold'] and row['similarity_info'] != 'None':
            try:
                similarity_data = json.loads(row['similarity_info'])
                for match in similarity_data:
                    detailed_data.append({
                        'nodeid': row['nodeid'],
                        'query_name': row['name'],
                        'alphafold_id': match['alphafold_id'],
                        'uniprot_id': match['uniprot_id'],
                        'relevance_score': match['relevance_score'],
                        'relevance_level': match['relevance_level'],
                        'protein_name': match.get('protein_name', 'N/A'),
                        'gene_name': match.get('gene_name', 'N/A'),
                        'organism': match.get('organism', 'N/A'),
                        'plddt_score': match.get('plddt_score', 'N/A'),
                        'confidence_level': match.get('confidence_level', 'N/A'),
                        'sequence_length': match.get('sequence_length', 0)
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
    """CLI entry: name-based AlphaFold resolution (paths relative to repo root)."""
    csv_file_path = "data/db_export_proteins.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return
    
    output_file = "results/uniprot_sequence_results.csv"
    max_results = 10
    
    print(f"\nSearch settings:")
    print(f"  Input CSV: {csv_file_path}")
    print(f"  Output CSV: {output_file}")
    print(f"  Max results per query: {max_results}")
    print("=" * 60)
    
    results = process_alphafold_search(
        csv_file_path=csv_file_path,
        output_file=output_file,
        max_results=max_results
    )
    
    if results is not None:
        total = len(results)
        found = results['found_in_alphafold'].sum()
        
        print(f"\nAlphaFold search summary:")
        print(f"  Total rows: {total}")
        print(f"  Found in AlphaFold: {found} ({found/total*100:.1f}%)")
        
        if found > 0:
            create_alphafold_detailed_report(results)
            
            avg_relevance = results[results['max_relevance'] > 0]['max_relevance'].mean()
            exact_matches = len(results[results['max_relevance'] == 100])
            
            print(f"  Mean relevance: {avg_relevance:.1f}%")
            print(f"  Exact name matches (100): {exact_matches}")

if __name__ == "__main__":
    main()
