import os
import time
import json
from typing import Dict, List, Optional

import requests
import pandas as pd


class UniProtSequenceSearcher:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org"
        self.search_url = "https://rest.uniprot.org/uniprotkb/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
    
    def search_by_sequence(self, sequence: str, max_results: int = 10) -> Dict:
        """
        Find proteins with similar sequences through a BLAST-like approach
        """
        try:
            # Clean the sequence
            sequence = sequence.strip().upper()
            
            # Check if the sequence is valid
            if not self._is_valid_sequence(sequence):
                return {
                    'success': True,
                    'found': False,
                    'count': 0,
                    'uniprot_ids': [],
                    'detailed_results': [],
                    'error': f"Invalid sequence: contains non-amino acid characters"
                }
            
            # Skip too short sequences
            if len(sequence) < 5:
                return {
                    'success': True,
                    'found': False,
                    'count': 0,
                    'uniprot_ids': [],
                    'detailed_results': [],
                    'error': f"Sequence too short ({len(sequence)} amino acids)"
                }
            
            # Search for proteins with different strategies
            results = self._search_with_multiple_strategies(sequence, max_results)
            
            if not results:
                return {
                    'success': True,
                    'found': False,
                    'count': 0,
                    'uniprot_ids': [],
                    'detailed_results': [],
                    'error': "No similar proteins found"
                }
            
            # Process the found results
            return self._process_search_results(sequence, results)
            
        except Exception as e:
            return {
                'success': False,
                'found': False,
                'count': 0,
                'uniprot_ids': [],
                'detailed_results': [],
                'error': f"Search error: {str(e)}"
            }
    
    def _is_valid_sequence(self, sequence: str) -> bool:
        """Check if the sequence is a valid amino acid sequence"""
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')  # Standard amino acids
        return all(aa in valid_aa for aa in sequence)
    
    def _search_with_multiple_strategies(self, sequence: str, max_results: int) -> List[Dict]:
        """
        Search with multiple strategies to find similar sequences
        """
        all_results = []
        
        # Strategy 1: Exact match search
        exact_results = self._search_exact_match(sequence, max_results)
        all_results.extend(exact_results)
        
        # Strategy 2: Substring search (if the sequence is not too long)
        if len(sequence) <= 50 and len(all_results) < max_results:
            substring_results = self._search_substring_matches(sequence, max_results - len(all_results))
            all_results.extend(substring_results)
        
        # Strategy 3: Search for similar proteins by keywords
        if len(all_results) < max_results:
            keyword_results = self._search_similar_by_keywords(sequence, max_results - len(all_results))
            all_results.extend(keyword_results)
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for result in all_results:
            uniprot_id = result.get('primaryAccession')
            if uniprot_id and uniprot_id not in seen_ids:
                seen_ids.add(uniprot_id)
                unique_results.append(result)
        
        return unique_results[:max_results]
    
    def _search_exact_match(self, sequence: str, max_results: int) -> List[Dict]:
        """Search for exact match of the sequence"""
        query = f'sequence:"{sequence}"'
        params = {
            'query': query,
            'fields': 'accession,protein_name,gene_names,organism_name,sequence',
            'size': max_results,
            'format': 'json'
        }
        
        try:
            print("Exact match search...")
            response = self.session.get(self.search_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                print(f"  Found {len(results)} exact matches")
                return results
        except Exception as e:
            print(f"  Exact search error: {e}")
        finally:
            return []
    
    def _search_substring_matches(self, sequence: str, max_results: int) -> List[Dict]:
        """Search for partial matches through substrings"""
        # For partial matches, use substrings of varying length.
        substrings = self._generate_search_substrings(sequence)
        
        all_results = []
        for substring in substrings:
            if len(all_results) >= max_results:
                break
                
            query = f'sequence:"{substring}"'
            params = {
                'query': query,
                'fields': 'accession,protein_name,gene_names,organism_name,sequence',
                'size': min(10, max_results - len(all_results)),
                'format': 'json'
            }
            
            try:
                print(f"  Substring search: {substring}...")
                response = self.session.get(self.search_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    for result in results:
                        if result not in all_results:
                            all_results.append(result)
                    
                    time.sleep(0.5)
            except Exception as e:
                print(f"  Substring search error: {e}")
                continue
        
        print(f"  Found {len(all_results)} substring matches")
        return all_results
    
    def _generate_search_substrings(self, sequence: str) -> List[str]:
        """Generate substrings for searching partial matches"""
        substrings = []
        seq_len = len(sequence)
        
        # For short sequences we use the entire sequence
        if seq_len <= 15:
            substrings.append(sequence)
        else:
            # For long sequences we take different parts
            # Start of the sequence
            substrings.append(sequence[:15])
            # End of the sequence
            substrings.append(sequence[-15:])
            # Middle of the sequence
            if seq_len > 30:
                mid_start = (seq_len - 15) // 2
                substrings.append(sequence[mid_start:mid_start + 15])
            # Most characteristic parts (first 10 and last 10)
            substrings.append(sequence[:10])
            substrings.append(sequence[-10:])
        
        return substrings
    
    def _search_similar_by_keywords(self, sequence: str, max_results: int) -> List[Dict]:
        """Search for similar proteins by keywords and filter by similarity"""
        # Take random reviewed proteins and filter by similarity
        query = 'reviewed:true'
        params = {
            'query': query,
            'fields': 'accession,protein_name,gene_names,organism_name,sequence',
            'size': min(max_results * 3, 50),  # Take more for filtering
            'format': 'json'
        }
        
        try:
            print("  Similar proteins search...")
            response = self.session.get(self.search_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                all_results = data.get('results', [])
                
                # Filter results by similarity with our sequence
                filtered_results = self._filter_by_sequence_similarity(sequence, all_results, max_results)
                print(f"  Found {len(filtered_results)} similar proteins")
                return filtered_results
        except Exception as e:
            print(f"  Similar search error: {e}")
        finally:
            return []
    
    def _filter_by_sequence_similarity(self, query_sequence: str, results: List[Dict], max_results: int) -> List[Dict]:
        """Filter results by similarity of sequences"""
        scored_results = []
        
        for result in results:
            db_sequence = result.get('sequence', {}).get('value', '')
            if not db_sequence:
                continue
            
            # Calculate similarity of sequences
            similarity_score = self._calculate_sequence_similarity(query_sequence, db_sequence)
            
            # Add only if similarity is above the threshold
            if similarity_score >= 0.3:  # Minimum similarity threshold: 30%
                scored_results.append((similarity_score, result))
        
        # Sort by similarity
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [result for score, result in scored_results[:max_results]]
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate similarity between two sequences
        Returns a value from 0.0 to 1.0
        """
        # Use the algorithm for finding the best local alignment
        best_similarity = 0.0
        
        # For short sequence we search in the long one
        if len(seq1) <= len(seq2):
            for i in range(0, len(seq2) - len(seq1) + 1):
                substring = seq2[i:i + len(seq1)]
                matches = sum(1 for a, b in zip(seq1, substring) if a == b)
                similarity = matches / len(seq1)
                if similarity > best_similarity:
                    best_similarity = similarity
        else:
            # For long sequence we search in the short one
            for i in range(0, len(seq1) - len(seq2) + 1):
                substring = seq1[i:i + len(seq2)]
                matches = sum(1 for a, b in zip(substring, seq2) if a == b)
                similarity = matches / len(seq2)
                if similarity > best_similarity:
                    best_similarity = similarity
        
        return best_similarity
    
    def _process_search_results(self, query_sequence: str, results: List[Dict]) -> Dict:
        """Process the search results"""
        uniprot_ids = []
        detailed_results = []
        
        for result in results:
            uniprot_id = result.get('primaryAccession')
            if not uniprot_id or uniprot_id in uniprot_ids:
                continue
                
            uniprot_ids.append(uniprot_id)
            
            # Get information about the protein
            protein_name = self._extract_protein_name(result)
            gene_names = self._extract_gene_names(result)
            db_sequence = result.get('sequence', {}).get('value', '')
            
            # Calculate detailed similarity
            similarity_info = self._calculate_detailed_similarity(query_sequence, db_sequence)
            
            detailed_results.append({
                'uniprot_id': uniprot_id,
                'identity_percent': similarity_info['identity_percent'],
                'similarity_level': self._get_similarity_level(similarity_info['identity_percent']),
                'protein_name': protein_name,
                'gene_name': ', '.join(gene_names),
                'organism': result.get('organism', {}).get('scientificName', 'N/A'),
                'length': result.get('sequence', {}).get('length', 0),
                'alignment_length': similarity_info['alignment_length'],
                'match_type': similarity_info['match_type'],
                'original_sequence': query_sequence,
                'found_sequence': db_sequence,
                'coverage': similarity_info['coverage']
            })
        
        # Sort by identity
        detailed_results.sort(key=lambda x: x['identity_percent'], reverse=True)
        
        return {
            'success': True,
            'found': len(detailed_results) > 0,
            'count': len(detailed_results),
            'uniprot_ids': uniprot_ids,
            'detailed_results': detailed_results,
            'error': None
        }
    
    def _calculate_detailed_similarity(self, query_seq: str, db_seq: str) -> Dict:
        """Detailed calculation of similarity between sequences"""
        if not db_seq:
            return {
                'identity_percent': 0.0,
                'alignment_length': 0,
                'match_type': 'no_match',
                'coverage': 0.0
            }
        
        # Check exact match
        if query_seq == db_seq:
            return {
                'identity_percent': 100.0,
                'alignment_length': len(query_seq),
                'match_type': 'exact_match',
                'coverage': 100.0
            }
        
        # Search for the best local alignment
        best_identity = 0.0
        best_length = 0
        best_match_type = 'partial'
        
        if len(query_seq) <= len(db_seq):
            # Search for substring occurrence
            for i in range(0, len(db_seq) - len(query_seq) + 1):
                substring = db_seq[i:i + len(query_seq)]
                matches = sum(1 for a, b in zip(query_seq, substring) if a == b)
                identity = (matches / len(query_seq)) * 100
                if identity > best_identity:
                    best_identity = identity
                    best_length = len(query_seq)
                    best_match_type = 'substring' if identity == 100 else 'partial'
        else:
            # Our sequence is longer - search for protein occurrence in our sequence
            for i in range(0, len(query_seq) - len(db_seq) + 1):
                substring = query_seq[i:i + len(db_seq)]
                matches = sum(1 for a, b in zip(substring, db_seq) if a == b)
                identity = (matches / len(db_seq)) * 100
                if identity > best_identity:
                    best_identity = identity
                    best_length = len(db_seq)
                    best_match_type = 'contains_protein'
        
        coverage = (min(len(query_seq), len(db_seq)) / max(len(query_seq), len(db_seq))) * 100
        
        return {
            'identity_percent': round(best_identity, 2),
            'alignment_length': best_length,
            'match_type': best_match_type,
            'coverage': round(coverage, 2)
        }
    
    def _get_similarity_level(self, percent: float) -> str:
        """Determine the similarity level"""
        if percent == 100:
            return "exact_match"
        elif percent >= 90:
            return "very_high"
        elif percent >= 80:
            return "high"
        elif percent >= 70:
            return "medium"
        elif percent >= 50:
            return "low"
        else:
            return "very_low"
    
    def _extract_protein_name(self, result: Dict) -> str:
        """Extract the protein name from the result"""
        try:
            protein_desc = result.get('proteinDescription', {})
            if protein_desc and 'recommendedName' in protein_desc:
                name_obj = protein_desc['recommendedName']
                if 'fullName' in name_obj:
                    return name_obj['fullName'].get('value', 'N/A')
        except:
            pass
        
        return 'N/A'
    
    def _extract_gene_names(self, result: Dict) -> List[str]:
        """Extract the gene names"""
        gene_names = []
        try:
            genes = result.get('genes', [])
            for gene in genes:
                if 'geneName' in gene:
                    name = gene['geneName'].get('value', '')
                    if name:
                        gene_names.append(name)
        except:
            pass
        return gene_names


def process_uniprot_search_by_sequence(
        csv_file_path: str,
        output_file: Optional[str] = None,
        max_results: int = 10
) -> pd.DataFrame:
    """
    Process the dataset with searching for sequence in UniProt
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: File {csv_file_path} not found!")
        return None
    
    df = pd.read_csv(csv_file_path)
    
    # Validate required columns.
    required_columns = ['nodeid', 'name', 'content']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' is missing in the dataset")
            return None
    
    print(f"Loaded records: {len(df)}")
    print("Searching in UniProt by sequence...")
    print("=" * 60)
    
    searcher = UniProtSequenceSearcher()
    results = []
    
    for index, row in df.iterrows():
        print(f"\nProcessing {index + 1}/{len(df)}: {row['name']}")
        sequence = row['content'].strip()
        print(f"Sequence: {sequence[:30]}...")
        print(f"Length: {len(sequence)} amino acids")
        
        result = searcher.search_by_sequence(sequence, max_results)
        
        similarity_info = result['detailed_results']
        
        # Form the result in the needed format
        result_data = {
            'nodeid': row['nodeid'],
            'name': row['name'],
            'sequence': sequence,
            'sequence_length': len(sequence),
            'found_in_uniprot': result['found'],
            'uniprot_results_count': result['count'],
            'uniprot_ids': ', '.join(result['uniprot_ids']) if result['uniprot_ids'] else 'None',
            'similarity_info': json.dumps(similarity_info, ensure_ascii=False) if similarity_info else 'None',
            'max_identity': max([item['identity_percent'] for item in similarity_info]) if similarity_info else 0,
            'best_match': similarity_info[0]['uniprot_id'] if similarity_info else 'None',
            'error': result['error'] if result['error'] else 'None'
        }
        
        results.append(result_data)
        
        status = "FOUND" if result['found'] else "NOT FOUND"
        print(f"  {status} - {result['count']} results")
        
        if result['found'] and similarity_info:
            best_match = similarity_info[0]
            print(f"  Best match: {best_match['uniprot_id']} ({best_match['identity_percent']}% identity)")
            print(f"  Protein name: {best_match['protein_name']}")
        elif result['error']:
            print(f"  Error: {result['error']}")
        
        time.sleep(2)  # Pause between requests to avoid blocking
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved to: {output_file}")
    
    return results_df

# Other functions (create_detailed_report, main) remain unchanged
def create_detailed_report(results_df: pd.DataFrame, output_file: str = "uniprot_detailed_report.csv") -> pd.DataFrame:
    """
    Create a detailed report
    """
    detailed_data = []
    
    for _, row in results_df.iterrows():
        if row['found_in_uniprot'] and row['similarity_info'] != 'None':
            try:
                similarity_data = json.loads(row['similarity_info'])
                for match in similarity_data:
                    detailed_data.append({
                        'nodeid': row['nodeid'],
                        'query_name': row['name'],
                        'uniprot_id': match['uniprot_id'],
                        'identity_percent': match['identity_percent'],
                        'similarity_level': match['similarity_level'],
                        'protein_name': match.get('protein_name', 'N/A'),
                        'gene_name': match.get('gene_name', 'N/A'),
                        'organism': match.get('organism', 'N/A'),
                        'length': match.get('length', 0),
                        'function': match.get('function', 'N/A'),
                        'original_sequence': match.get('original_sequence', 'N/A'),
                        'found_sequence': match.get('found_sequence', 'N/A')
                    })
            except Exception as e:
                print(f"Error processing detailed data: {e}")
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Detailed report saved to: {output_file}")
        return detailed_df
    else:
        print("No data for detailed report")
        return None

def main():
    """
    Main function
    """
    # csv_file_path = input("Enter path to input CSV: ").strip()
    csv_file_path = "data/db_export_proteins.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"File {csv_file_path} not found!")
        return
    
    # output_file = input("Enter the name for the output file (or press Enter for 'uniprot_sequence_results.csv'): ").strip()
    # if not output_file:
    output_file = "results/uniprot_sequence_results.csv"
    
    # try:
    #     max_results = int(input("Enter the maximum number of results per request (default 10): ").strip() or "10")
    # except ValueError:
    #     max_results = 10

    max_results = 10
    
    print(f"\nSearch settings:")
    print(f"  Data file: {csv_file_path}")
    print(f"  Output file: {output_file}")
    print(f"  Maximum results: {max_results}")
    print("=" * 60)
    
    # Start the search by sequence
    results = process_uniprot_search_by_sequence(
        csv_file_path=csv_file_path,
        output_file=output_file,
        max_results=max_results
    )
    
    if results is not None:
        # Statistics
        total = len(results)
        found = results['found_in_uniprot'].sum()
        
        print(f"\nSearch statistics:")
        print(f"  Total records: {total}")
        print(f"  Found in UniProt: {found} ({found/total*100:.1f}%)")
        
        if found > 0:
            # Create a detailed report
            detailed_report = create_detailed_report(results)
            
            # Additional statistics
            avg_identity = results[results['max_identity'] > 0]['max_identity'].mean()
            exact_matches = len(results[results['max_identity'] == 100])
            print(f"  Average identity: {avg_identity:.1f}%")
            print(f"  Exact matches: {exact_matches}")

if __name__ == "__main__":
    main()
