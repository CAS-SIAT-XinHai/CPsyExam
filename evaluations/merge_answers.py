import argparse
import json
import os
from collections import defaultdict

def detect_n_shot(result_dir):
    """Detect n_shot by reading the first result file in the directory"""
    for filename in os.listdir(result_dir):
        if filename.startswith('results_cpsyexam_') and filename.endswith('.json'):
            with open(os.path.join(result_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data['results'] and 'n_shot' in data['results'][0]:
                    return data['results'][0]['n_shot']
    return None

def load_results(result_dir):
    """Load all result files from a directory"""
    results = {}
    for filename in os.listdir(result_dir):
        if filename.startswith('results_cpsyexam_') and filename.endswith('.json'):
            with open(os.path.join(result_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract subject name from filename
                subject = filename.replace('results_cpsyexam_', '').replace('.json', '')
                results[subject] = data['results']
    return results

def merge_results(result_dirs, output_file):
    """Merge results from multiple directories and save to a single file"""
    if len(result_dirs) != 2:
        raise ValueError("Exactly two directories must be provided")
    
    # Detect n_shot for each directory
    dir_shots = []
    for result_dir in result_dirs:
        n_shot = detect_n_shot(result_dir)
        if n_shot is None:
            raise ValueError(f"Could not detect n_shot in directory: {result_dir}")
        dir_shots.append((result_dir, n_shot))
    
    # Sort directories by n_shot (zero-shot first, then five-shot)
    dir_shots.sort(key=lambda x: x[1])
    
    if dir_shots[0][1] != 0 or dir_shots[1][1] != 5:
        raise ValueError(f"Expected one zero-shot and one five-shot directory, got: {[shot for _, shot in dir_shots]}")
    
    # Load results from sorted directories
    zero_shot_results = load_results(dir_shots[0][0])  # n_shot = 0
    five_shot_results = load_results(dir_shots[1][0])  # n_shot = 5
    
    # Create merged structure with separate zero and five shot sections
    merged_results = {
        "zero_shot": zero_shot_results,
        "five_shot": five_shot_results
    }
    
    # Save merged results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\nMerge Summary:")
    print(f"Zero-shot results (from {os.path.basename(dir_shots[0][0])}):")
    for subject in zero_shot_results:
        print(f"  {subject}: {len(zero_shot_results[subject])} predictions")
    print(f"\nFive-shot results (from {os.path.basename(dir_shots[1][0])}):")
    for subject in five_shot_results:
        print(f"  {subject}: {len(five_shot_results[subject])} predictions")
    
    return merged_results

def main():
    parser = argparse.ArgumentParser(description='Merge CPsyExam evaluation results')
    parser.add_argument('--result_dirs', nargs=2, required=True,
                        help='Two directories containing results (order does not matter)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save merged results')
    args = parser.parse_args()
    
    # Merge results
    merged_results = merge_results(args.result_dirs, args.output_file)
    print(f"\nMerged results saved to: {args.output_file}")

if __name__ == '__main__':
    main() 