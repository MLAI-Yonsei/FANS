#!/usr/bin/env python3
"""
Script to analyze FANS results across all experiments and calculate F1 scores.
Focuses on distinguishing function_only vs function_and_noise shifts.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def parse_directory_name(dir_name: str) -> Tuple[int, str, int]:
    """
    Parse directory name to extract nodes, graph_type, and adj_num.
    
    Examples:
        - mlainas_statduck_data_nodes_10_SF_adj_5 -> (10, 'SF', 5)
        - nodes_20_ER_adj_3 -> (20, 'ER', 3)
    """
    # Pattern with prefix
    pattern1 = r'nodes_(\d+)_(ER|SF)_adj_(\d+)'
    match = re.search(pattern1, dir_name)
    
    if match:
        nodes = int(match.group(1))
        graph_type = match.group(2)
        adj_num = int(match.group(3))
        return nodes, graph_type, adj_num
    
    raise ValueError(f"Could not parse directory name: {dir_name}")


def load_metadata(nodes: int, graph_type: str, adj_num: int) -> Dict:
    """Load metadata from the data directory."""
    # Try multiple possible paths
    possible_paths = [
        f"/mlainas/statduck/data_both/data_small/nodes_{nodes}/{graph_type}/metadata_{adj_num}.json",
        f"/home/statduck/causal-flows/data_both/data/nodes_{nodes}/{graph_type}/metadata_{adj_num}.json",
        f"/home/statduck/causal-flows/data_both/data_small/nodes_{nodes}/{graph_type}/metadata_{adj_num}.json",
    ]
    
    for metadata_path in possible_paths:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"Metadata not found for nodes_{nodes}/{graph_type}/metadata_{adj_num}.json")


def load_fans_results(fans_json_path: str) -> Dict:
    """Load FANS results from JSON file."""
    with open(fans_json_path, 'r') as f:
        return json.load(f)


def has_simultaneous_shift_files(fans_analysis_dir: Path) -> bool:
    """
    Check if fans_analysis directory contains files starting with 'simultaneous_'.
    
    Args:
        fans_analysis_dir: Path to fans_analysis directory
        
    Returns:
        True if at least one file starting with 'simultaneous_' exists
    """
    if not fans_analysis_dir.exists() or not fans_analysis_dir.is_dir():
        return False
    
    for file in fans_analysis_dir.iterdir():
        if file.is_file() and file.name.startswith('simultaneous_'):
            return True
    
    return False


def extract_shift_info(metadata: Dict) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract shift information from metadata.
    
    Returns:
        shifted_nodes: List of all shifted node indices
        true_function_only: List of nodes with ONLY function shifts
        true_function_and_noise: List of nodes with function + noise shifts
    """
    shifted_nodes = metadata.get('shifted_nodes', [])
    shift_types = metadata.get('shift_types', {})
    
    true_function_only = []
    true_function_and_noise = []
    
    for node_idx in shifted_nodes:
        node_key = str(node_idx)
        if node_key in shift_types:
            shift_list = shift_types[node_key]
            
            # Check if it has function_shift
            has_function = 'function_shift' in shift_list
            # Check if it has noise shift (variance_shift or distribution_shift)
            has_noise = ('variance_shift' in shift_list) or ('distribution_shift' in shift_list)
            
            if has_function and not has_noise:
                true_function_only.append(node_idx)
            elif has_function and has_noise:
                true_function_and_noise.append(node_idx)
    
    return shifted_nodes, true_function_only, true_function_and_noise


def extract_detection_info(fans_results: Dict) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract detection information from FANS simultaneous_shift_results.
    
    Returns:
        analyzed_nodes: List of analyzed nodes
        detected_function_only: List of nodes detected as function_only
        detected_function_and_noise: List of nodes detected as function_and_noise
    """
    analyzed_nodes = fans_results.get('analyzed_nodes', [])
    simultaneous_shift_results = fans_results.get('simultaneous_shift_results', {})
    
    detected_function_only = []
    detected_function_and_noise = []
    
    if simultaneous_shift_results is None:
        # If simultaneous_shift was not run, return empty lists
        return analyzed_nodes, [], []
    
    for node_idx in analyzed_nodes:
        node_key = str(node_idx)
        if node_key in simultaneous_shift_results:
            shift_type = simultaneous_shift_results[node_key].get('shift_type', None)
            
            if shift_type == 'function_only':
                detected_function_only.append(node_idx)
            elif shift_type == 'function_and_noise':
                detected_function_and_noise.append(node_idx)
    
    return analyzed_nodes, detected_function_only, detected_function_and_noise


def find_experiment_folders(output_testing_dir: str) -> List[Tuple[str, str]]:
    """
    Find all experiment folders containing fans_results.json with simultaneous shift analysis.
    Only selects exp_dirs that have files starting with 'simultaneous_' in fans_analysis.
    
    Returns:
        List of tuples: (parent_dir_name, full_path_to_fans_results.json)
    """
    results = []
    output_path = Path(output_testing_dir)
    
    # Iterate through directories matching the pattern
    for parent_dir in output_path.iterdir():
        if not parent_dir.is_dir():
            continue
        
        # Skip directories that don't match the expected pattern
        try:
            nodes, graph_type, adj_num = parse_directory_name(parent_dir.name)
        except ValueError:
            continue
        
        # Look for exp_dirs with simultaneous shift analysis
        found_valid_exp = False
        for exp_dir in parent_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            fans_analysis_dir = exp_dir / 'fans_analysis'
            fans_json = fans_analysis_dir / 'fans_results.json'
            
            # Check if fans_results.json exists and has simultaneous shift files
            if fans_json.exists() and has_simultaneous_shift_files(fans_analysis_dir):
                results.append((parent_dir.name, str(fans_json)))
                print(f"Selected {exp_dir.name} for {parent_dir.name} (has simultaneous shift analysis)")
                found_valid_exp = True
                break  # Found valid exp_dir, move to next parent_dir
        
        if not found_valid_exp:
            print(f"No valid experiment found for {parent_dir.name} (missing simultaneous shift analysis)")
    
    return results


def create_results_dataframe(output_testing_dir: str) -> pd.DataFrame:
    """
    Create a comprehensive dataframe with all results.
    """
    experiment_folders = find_experiment_folders(output_testing_dir)
    
    print(f"\nFound {len(experiment_folders)} experiment folders with simultaneous shift analysis")
    
    rows = []
    
    for parent_dir_name, fans_json_path in experiment_folders:
        try:
            # Parse directory information
            nodes, graph_type, adj_num = parse_directory_name(parent_dir_name)
            
            # Load metadata
            metadata = load_metadata(nodes, graph_type, adj_num)
            
            # Load FANS results
            fans_results = load_fans_results(fans_json_path)
            
            # Extract information
            shifted_nodes, true_function_only, true_function_and_noise = extract_shift_info(metadata)
            analyzed_nodes, detected_function_only, detected_function_and_noise = extract_detection_info(fans_results)
            
            # Create row
            row = {
                'nodes': nodes,
                'graph_type': graph_type,
                'adj_num': adj_num,
                'shifted_nodes': shifted_nodes,
                'analyzed_nodes': analyzed_nodes,
                'true_function_only': true_function_only,
                'true_function_and_noise': true_function_and_noise,
                'detected_function_only': detected_function_only,
                'detected_function_and_noise': detected_function_and_noise,
            }
            
            rows.append(row)
            
        except Exception as e:
            print(f"Error processing {parent_dir_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['nodes', 'graph_type', 'adj_num']).reset_index(drop=True)
    
    return df


def calculate_f1_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate F1 scores for function_only vs function_and_noise classification.
    Uses the same logic as analysis.py for consistent F1 calculation.
    """
    results = []
    
    # Group by nodes and graph_type
    for (nodes, graph_type), group in df.groupby(['nodes', 'graph_type']):
        # Initialize counters for manual F1 calculation
        tp_function = 0
        fn_function = 0
        tp_noise = 0
        fn_noise = 0
        
        for idx, row in group.iterrows():
            # Only consider analyzed nodes that were actually shifted
            analyzed_shifted = [n for n in row['analyzed_nodes'] if n in row['shifted_nodes']]
            
            # Get true and predicted sets
            true_function = set(row['true_function_only'])
            true_noise = set(row['true_function_and_noise'])
            pred_function = set(row['detected_function_only'])
            pred_noise = set(row['detected_function_and_noise'])
            
            for node in analyzed_shifted:
                # True function nodes
                if node in true_function:
                    if node in pred_function:
                        tp_function += 1
                    else:
                        fn_function += 1
                # True noise nodes
                elif node in true_noise:
                    if node in pred_noise:
                        tp_noise += 1
                    else:
                        fn_noise += 1
        
        # FP calculation: misclassifying one class as the other
        fp_function = fn_noise
        fp_noise = fn_function
        
        # Calculate F1 for function class
        if tp_function + fp_function + fn_function > 0:
            precision_func = tp_function / max(tp_function + fp_function, 1)
            recall_func = tp_function / max(tp_function + fn_function, 1)
            if precision_func + recall_func > 0:
                f1_function = 2 * (precision_func * recall_func) / (precision_func + recall_func)
            else:
                f1_function = 0.0
        else:
            f1_function = 0.0
        
        # Calculate F1 for noise class
        if tp_noise + fp_noise + fn_noise > 0:
            precision_noise = tp_noise / max(tp_noise + fp_noise, 1)
            recall_noise = tp_noise / max(tp_noise + fn_noise, 1)
            if precision_noise + recall_noise > 0:
                f1_noise = 2 * (precision_noise * recall_noise) / (precision_noise + recall_noise)
            else:
                f1_noise = 0.0
        else:
            f1_noise = 0.0
        
        # Calculate macro F1
        if (tp_function + fn_function > 0) and (tp_noise + fn_noise > 0):
            # Both classes exist
            f1_macro = (f1_function + f1_noise) / 2.0
        elif tp_function + fn_function > 0:
            # Only function class exists
            f1_macro = f1_function
        elif tp_noise + fn_noise > 0:
            # Only noise class exists
            f1_macro = f1_noise
        else:
            f1_macro = 0.0
        
        # Calculate accuracy
        total = tp_function + fn_function + tp_noise + fn_noise
        correct = tp_function + tp_noise
        accuracy = correct / max(total, 1)
        
        results.append({
            'nodes': nodes,
            'graph_type': graph_type,
            'num_experiments': len(group),
            'total_classified': total,
            'correct_classified': correct,
            'accuracy': accuracy,
            'f1_function': f1_function,
            'f1_noise': f1_noise,
            'f1_macro': f1_macro,
        })
    
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    output_testing_dir = "/home/statduck/fans/experiments/results/both/exp2_fans"
    
    print("=" * 80)
    print("FANS Results Analysis - Function Only vs Function + Noise")
    print("=" * 80)
    
    # Create detailed results dataframe
    print("\nCreating detailed results dataframe...")
    df_detailed = create_results_dataframe(output_testing_dir)
    
    print(f"\nTotal experiments processed: {len(df_detailed)}")
    print(f"\nBreakdown by nodes and graph type:")
    print(df_detailed.groupby(['nodes', 'graph_type']).size())
    
    # Display sample of detailed results
    print("\nSample of detailed results:")
    for idx, row in df_detailed.head(3).iterrows():
        print(f"\nExperiment: nodes={row['nodes']}, graph_type={row['graph_type']}, adj_num={row['adj_num']}")
        print(f"  Shifted nodes: {row['shifted_nodes']}")
        print(f"  True function_only: {row['true_function_only']}")
        print(f"  True function_and_noise: {row['true_function_and_noise']}")
        print(f"  Detected function_only: {row['detected_function_only']}")
        print(f"  Detected function_and_noise: {row['detected_function_and_noise']}")
    
    # Save detailed results
    detailed_csv = "/home/statduck/fans/experiments/analysis/fans_detailed_results.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"\nDetailed results saved to: {detailed_csv}")
    
    # Calculate F1 scores
    print("\nCalculating F1 scores...")
    df_f1 = calculate_f1_scores(df_detailed)
    
    print("\n" + "=" * 80)
    print("Classification Performance: Function Only vs Function + Noise")
    print("=" * 80)
    print(df_f1.to_string(index=False))
    
    # Save F1 scores
    f1_csv = "/home/statduck/fans/experiments/analysis/fans_f1_scores.csv"
    df_f1.to_csv(f1_csv, index=False)
    print(f"\nF1 scores saved to: {f1_csv}")
    
    # Additional statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"\nAverage Accuracy: {df_f1['accuracy'].mean():.4f}")
    print(f"Average F1 Function: {df_f1['f1_function'].mean():.4f}")
    print(f"Average F1 Noise: {df_f1['f1_noise'].mean():.4f}")
    print(f"Average F1 Macro: {df_f1['f1_macro'].mean():.4f}")
    print(f"Total nodes classified: {df_f1['total_classified'].sum()}")
    print(f"Total correct: {df_f1['correct_classified'].sum()}")
    
    # Breakdown by nodes
    print("\n\nBy number of nodes:")
    print(df_f1.groupby('nodes')[['accuracy', 'f1_function', 'f1_noise', 'f1_macro', 'total_classified']].agg({
        'accuracy': 'mean',
        'f1_function': 'mean',
        'f1_noise': 'mean',
        'f1_macro': 'mean',
        'total_classified': 'sum'
    }))
    
    # Breakdown by graph type
    print("\n\nBy graph type:")
    print(df_f1.groupby('graph_type')[['accuracy', 'f1_function', 'f1_noise', 'f1_macro', 'total_classified']].agg({
        'accuracy': 'mean',
        'f1_function': 'mean',
        'f1_noise': 'mean',
        'f1_macro': 'mean',
        'total_classified': 'sum'
    }))


if __name__ == "__main__":
    main()