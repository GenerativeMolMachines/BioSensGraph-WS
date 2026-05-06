import requests
import pandas as pd
import time
import json
import os

def search_pdb_by_sequence_rcsb(sequence, identity_cutoff=0.9, max_results=10):
    """
    Search PDB via the RCSB Search API sequence service.

    Args:
        sequence (str): Amino-acid sequence.
        identity_cutoff (float): Identity threshold (0.0–1.0).
        max_results (int): Maximum hits to return.

    Returns:
        dict: Match metadata and raw identifiers.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity_cutoff,
                "target": "pdb_protein_sequence",
                "value": sequence
            }
        },
        "request_options": {
            "scoring_strategy": "sequence",
            "paginate": {
                "start": 0,
                "rows": max_results
            }
        },
        "return_type": "polymer_entity"
    }
    
    try:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.post(url, json=query, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        result_set = data.get('result_set', [])
        
        detailed_results = []
        for result in result_set:
            if isinstance(result, dict):
                pdb_info = {
                    'identifier': result.get('identifier', ''),
                    'score': result.get('score', 0),
                    'similarity': round(result.get('score', 0) * 100, 2)
                }
                detailed_results.append(pdb_info)
        
        pdb_ids = [result.get('identifier', '') for result in detailed_results if result.get('identifier')]
        
        return {
            'success': True,
            'found': len(pdb_ids) > 0,
            'count': len(pdb_ids),
            'pdb_ids': pdb_ids,
            'detailed_results': detailed_results,
            'raw_result': result_set,
            'error': None
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'found': False,
            'count': 0,
            'pdb_ids': [],
            'detailed_results': [],
            'raw_result': [],
            'error': f"Request error: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'found': False,
            'count': 0,
            'pdb_ids': [],
            'detailed_results': [],
            'raw_result': [],
            'error': f"Unexpected error: {str(e)}"
        }

def get_pdb_info(pdb_id):
    """Fetch core entry metadata for a PDB ID."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_sequence_similarity_details(pdb_id, similarity_score):
    """Minimal similarity annotation (extend with more RCSB calls if needed)."""
    return {
        'pdb_id': pdb_id,
        'similarity_percent': similarity_score,
        'similarity_level': get_similarity_level(similarity_score),
        'match_type': 'homology' if similarity_score < 100 else 'exact'
    }

def get_similarity_level(percent):
    """Label bucket for similarity percentage."""
    if percent == 100:
        return "exact_match"
    elif percent >= 90:
        return "very_high"
    elif percent >= 80:
        return "high"
    elif percent >= 70:
        return "medium"
    elif percent >= 60:
        return "low"
    else:
        return "very_low"

def process_dataset_with_rcsb(csv_file_path, output_file=None, identity_cutoff=0.8):
    """Run RCSB sequence search for each row in a nodeid/name/content CSV."""
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
    print("PDB sequence search...")
    print("=" * 60)
    
    results = []
    
    for index, row in df.iterrows():
        print(f"\nRow {index + 1}/{len(df)}: {row['name']}")
        print(f"Sequence: {row['content'][:30]}...")
        print(f"Length: {len(row['content'])} residues")
        
        sequence = row['content']
        result = search_pdb_by_sequence_rcsb(sequence, identity_cutoff)
        
        similarity_info = []
        if result['found']:
            for pdb_result in result['detailed_results']:
                pdb_id = pdb_result.get('identifier', '')
                similarity_score = pdb_result.get('similarity', 0)
                
                info = get_pdb_info(pdb_id)
                pdb_details = {
                    'pdb_id': pdb_id,
                    'similarity_percent': similarity_score,
                    'similarity_level': get_similarity_level(similarity_score),
                    'title': info.get('struct', {}).get('title', 'N/A') if info else 'N/A',
                    'resolution': info.get('rcsb_entry_info', {}).get('resolution_combined', ['N/A'])[0] if info and info.get('rcsb_entry_info', {}).get('resolution_combined') else 'N/A',
                }
                similarity_info.append(pdb_details)
                
                time.sleep(0.5)
        
        result_data = {
            'nodeid': row['nodeid'],
            'name': row['name'],
            'sequence': sequence,
            'sequence_length': len(sequence),
            'found_in_pdb': result['found'],
            'pdb_results_count': result['count'],
            'pdb_ids': ', '.join(result['pdb_ids']) if result['pdb_ids'] else 'None',
            'similarity_info': json.dumps(similarity_info, ensure_ascii=False) if similarity_info else 'None',
            'max_similarity': max([item['similarity_percent'] for item in similarity_info]) if similarity_info else 0,
            'best_match': similarity_info[0]['pdb_id'] if similarity_info else 'None',
            'error': result['error'] if result['error'] else 'None'
        }
        
        results.append(result_data)
        
        status = "FOUND" if result['found'] else "NOT FOUND"
        print(f"  {status} — {result['count']} hit(s)")
        if result['found'] and similarity_info:
            best_match = similarity_info[0]
            print(f"  Best match: {best_match['pdb_id']} ({best_match['similarity_percent']}% similarity)")
            print(f"  Similarity level: {best_match['similarity_level']}")
        
        time.sleep(2)
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults written to: {output_file}")
    
    return results_df

def create_detailed_similarity_report(results_df, output_file="results/similarity_report.csv"):
    """Explode stored similarity JSON into a flat CSV."""
    detailed_data = []
    
    for _, row in results_df.iterrows():
        if row['found_in_pdb'] and row['similarity_info'] != 'None':
            try:
                similarity_data = json.loads(row['similarity_info'])
                for match in similarity_data:
                    detailed_data.append({
                        'nodeid': row['nodeid'],
                        'name': row['name'],
                        'query_sequence_length': row['sequence_length'],
                        'pdb_id': match['pdb_id'],
                        'similarity_percent': match['similarity_percent'],
                        'similarity_level': match['similarity_level'],
                        'title': match['title'],
                        'resolution': match['resolution']
                    })
            except:
                pass
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Detailed similarity report saved to: {output_file}")
        return detailed_df
    else:
        print("No rows for detailed similarity report")
        return None

def print_summary(results_df):
    """Print a text summary of PDB hit rates."""
    if results_df is None:
        return
    
    print("\n" + "="*80)
    print("PDB SEARCH SUMMARY")
    print("="*80)
    
    total = len(results_df)
    found = results_df['found_in_pdb'].sum()
    
    print(f"Sequences processed: {total}")
    print(f"Found in PDB: {found} ({found/total*100:.1f}%)")
    print(f"Not found: {total - found} ({(total-found)/total*100:.1f}%)")
    
    if found > 0:
        avg_similarity = results_df[results_df['max_similarity'] > 0]['max_similarity'].mean()
        max_similarity = results_df['max_similarity'].max()
        print(f"Mean best similarity: {avg_similarity:.1f}%")
        print(f"Max similarity: {max_similarity:.1f}%")
    
    print("\nPer-row results:")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        status = "FOUND" if row['found_in_pdb'] else "NOT FOUND"
        print(f"{row['nodeid']}: {row['name']} — {status}")
        if row['found_in_pdb']:
            print(f"  Hit count: {row['pdb_results_count']}")
            print(f"  Best similarity: {row['max_similarity']}%")
            print(f"  Best PDB ID: {row['best_match']}")
        print("-" * 40)

def main():
    """CLI: batch PDB homology search; default paths are relative to the repository root."""
    csv_file_path = "data/db_export_proteins.csv"

    if not os.path.exists(csv_file_path):
        print(f"File not found: {csv_file_path}")
        return
    
    output_file = "results/pdb_results.csv"
    identity_cutoff = 0.8
    
    print(f"\nSearch settings:")
    print(f"  Input CSV: {csv_file_path}")
    print(f"  Output CSV: {output_file}")
    print(f"  Identity cutoff: {identity_cutoff}")
    print("=" * 60)
    
    results = process_dataset_with_rcsb(
        csv_file_path=csv_file_path,
        output_file=output_file,
        identity_cutoff=identity_cutoff
    )
    
    if results is not None:
        print_summary(results)
        
        create_detailed_similarity_report(results, "results/detailed_similarity_report.csv")
        
        summary_file = "results/search_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PDB SEARCH SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input CSV: {csv_file_path}\n")
            f.write(f"Identity cutoff: {identity_cutoff}\n")
            f.write(f"Sequences: {len(results)}\n")
            f.write(f"Found in PDB: {results['found_in_pdb'].sum()}\n")
            f.write(f"Hit rate: {results['found_in_pdb'].sum()/len(results)*100:.1f}%\n\n")
            
            if results['found_in_pdb'].sum() > 0:
                avg_similarity = results[results['max_similarity'] > 0]['max_similarity'].mean()
                f.write(f"Mean similarity: {avg_similarity:.1f}%\n\n")
            
            f.write("Per-row results:\n")
            f.write("-" * 40 + "\n")
            for _, row in results.iterrows():
                status = "FOUND" if row['found_in_pdb'] else "NOT FOUND"
                f.write(f"{row['nodeid']}: {row['name']} — {status}\n")
                if row['found_in_pdb']:
                    f.write(f"  PDB IDs: {row['pdb_ids']}\n")
                    f.write(f"  Best similarity: {row['max_similarity']}%\n")
                f.write("\n")
        
        print(f"\nBrief summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
