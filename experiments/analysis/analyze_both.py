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

# Project root directory
SCRIPT_DIR = Path(__file__).parent
FANS_ROOT = SCRIPT_DIR.parent.parent  
DATA_DIR = FANS_ROOT / "data"
RESULTS_DIR = FANS_ROOT / "experiments" / "results"
ANALYSIS_DIR = FANS_ROOT / "experiments" / "analysis"


def parse_directory_name(dir_name: str) -> Tuple[int, str, int]:
    """
    Parse directory name to extract nodes, graph_type, and adj_num.
    
    Examples:
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
    # Try multiple possible paths within fans/data directory
    possible_paths = [
        DATA_DIR / "data_both" / "data_small" / f"nodes_{nodes}" / graph_type / f"metadata_{adj_num}.json",
        DATA_DIR / "data_both" / "data" / f"nodes_{nodes}" / graph_type / f"metadata_{adj_num}.json",
        DATA_DIR / "data_small" / f"nodes_{nodes}" / graph_type / f"metadata_{adj_num}.json",
    ]
    
    for metadata_path in possible_paths:
        if metadata_path.exists():
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


def calculate_f1_scores(df: pd.DataFrame, method_name: str = "FANS") -> pd.DataFrame:
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
            'method': method_name,
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


def find_gpr_files(gpr_dir: str, nodes: int = 10, graph_type: str = 'ER', 
                   start_idx: int = 1, end_idx: int = 30) -> List[str]:
    """
    Find GPR result JSON files in the specified directory.
    
    Args:
        gpr_dir: Base directory for GPR results
        nodes: Number of nodes
        graph_type: Graph type (ER or SF)
        start_idx: Starting index (inclusive)
        end_idx: Ending index (inclusive)
    
    Returns:
        List of full paths to GPR JSON files
    """
    gpr_files = []
    base_path = Path(gpr_dir) / f"nodes_{nodes}" / graph_type
    
    for idx in range(start_idx, end_idx + 1):
        json_file = base_path / f"gpr_optimized_nodes{nodes}_{graph_type}_{idx}_cpu.json"
        if json_file.exists():
            gpr_files.append(str(json_file))
        else:
            print(f"Warning: GPR file not found: {json_file}")
    
    return gpr_files


def load_gpr_result(gpr_json_path: str) -> Dict:
    """Load GPR result from JSON file."""
    with open(gpr_json_path, 'r') as f:
        return json.load(f)


def extract_gpr_detection_info(gpr_result: Dict) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract detection information from GPR results.
    
    Returns:
        analyzed_nodes: List of analyzed nodes (shifted nodes)
        detected_function_only: List of nodes detected as function_only
        detected_function_and_noise: List of nodes detected as function_and_noise
    """
    dataset_info = gpr_result.get('dataset_info', {})
    shifted_nodes = dataset_info.get('shifted_nodes', [])
    estimated_shift_types = gpr_result.get('estimated_shift_types', {})
    
    detected_function_only = []
    detected_function_and_noise = []
    
    for node_idx in shifted_nodes:
        node_key = str(node_idx)
        if node_key in estimated_shift_types:
            shift_type = estimated_shift_types[node_key]
            
            if shift_type == 'function':
                detected_function_only.append(node_idx)
            elif shift_type == 'simultaneous':
                detected_function_and_noise.append(node_idx)
    
    return shifted_nodes, detected_function_only, detected_function_and_noise


def create_gpr_results_dataframe(gpr_dir: str, nodes: int = 10, graph_type: str = 'ER',
                                 start_idx: int = 1, end_idx: int = 30) -> pd.DataFrame:
    """
    Create a comprehensive dataframe with GPR results.
    """
    gpr_files = find_gpr_files(gpr_dir, nodes, graph_type, start_idx, end_idx)
    
    print(f"\nFound {len(gpr_files)} GPR result files")
    
    rows = []
    
    for gpr_json_path in gpr_files:
        try:
            # Load GPR result
            gpr_result = load_gpr_result(gpr_json_path)
            
            # Extract dataset info (contains ground truth)
            dataset_info = gpr_result.get('dataset_info', {})
            nodes = dataset_info.get('node_count', nodes)
            graph_type = dataset_info.get('graph_type', graph_type)
            
            # Extract shift information (ground truth)
            shifted_nodes, true_function_only, true_function_and_noise = extract_shift_info(dataset_info)
            
            # Extract detection information (predictions)
            analyzed_nodes, detected_function_only, detected_function_and_noise = extract_gpr_detection_info(gpr_result)
            
            # Create row
            row = {
                'nodes': nodes,
                'graph_type': graph_type,
                'adj_num': dataset_info.get('dataset_index', -1),
                'shifted_nodes': shifted_nodes,
                'analyzed_nodes': analyzed_nodes,
                'true_function_only': true_function_only,
                'true_function_and_noise': true_function_and_noise,
                'detected_function_only': detected_function_only,
                'detected_function_and_noise': detected_function_and_noise,
            }
            
            rows.append(row)
            
        except Exception as e:
            print(f"Error processing {gpr_json_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['nodes', 'graph_type', 'adj_num']).reset_index(drop=True)
    
    return df


def create_comparison_table(df_fans_f1: pd.DataFrame, df_gpr_f1: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comparison table between FANS and GPR results.
    
    Args:
        df_fans_f1: F1 scores dataframe for FANS
        df_gpr_f1: F1 scores dataframe for GPR
    
    Returns:
        Comparison dataframe with side-by-side metrics
    """
    # Merge on nodes and graph_type
    comparison = pd.merge(
        df_fans_f1[['nodes', 'graph_type', 'num_experiments', 'total_classified', 
                    'correct_classified', 'accuracy', 'f1_function', 'f1_noise', 'f1_macro',
                    'precision_function', 'recall_function', 'precision_noise', 'recall_noise']],
        df_gpr_f1[['nodes', 'graph_type', 'num_experiments', 'total_classified', 
                   'correct_classified', 'accuracy', 'f1_function', 'f1_noise', 'f1_macro',
                   'precision_function', 'recall_function', 'precision_noise', 'recall_noise']],
        on=['nodes', 'graph_type'],
        suffixes=('_fans', '_gpr'),
        how='outer'
    )
    
    # Calculate differences
    comparison['accuracy_diff'] = comparison['accuracy_fans'] - comparison['accuracy_gpr']
    comparison['f1_macro_diff'] = comparison['f1_macro_fans'] - comparison['f1_macro_gpr']
    comparison['f1_function_diff'] = comparison['f1_function_fans'] - comparison['f1_function_gpr']
    comparison['f1_noise_diff'] = comparison['f1_noise_fans'] - comparison['f1_noise_gpr']
    
    return comparison


def main():
    """Main execution function."""
    output_testing_dir = str(RESULTS_DIR / "both" / "exp2_fans")
    gpr_dir = str(RESULTS_DIR / "both" /"exp2_gpr")
    
    print("=" * 80)
    print("FANS and GPR Results Analysis - Function Only vs Function + Noise")
    print("=" * 80)
    
    # ========== FANS Analysis ==========
    print("\n" + "=" * 80)
    print("ANALYZING FANS RESULTS")
    print("=" * 80)
    
    # Create detailed results dataframe
    print("\nCreating FANS detailed results dataframe...")
    df_fans_detailed = create_results_dataframe(output_testing_dir)
    
    print(f"\nTotal FANS experiments processed: {len(df_fans_detailed)}")
    print(f"\nBreakdown by nodes and graph type:")
    print(df_fans_detailed.groupby(['nodes', 'graph_type']).size())
    
    # Calculate FANS F1 scores
    print("\nCalculating FANS F1 scores...")
    df_fans_f1 = calculate_f1_scores(df_fans_detailed, method_name="FANS")
    
    # ========== GPR Analysis ==========
    print("\n" + "=" * 80)
    print("ANALYZING GPR RESULTS")
    print("=" * 80)
    
    # Create GPR results dataframe
    print("\nCreating GPR detailed results dataframe...")
    df_gpr_detailed = create_gpr_results_dataframe(
        gpr_dir=gpr_dir,
        nodes=10,
        graph_type='ER',
        start_idx=1,
        end_idx=30
    )
    
    print(f"\nTotal GPR experiments processed: {len(df_gpr_detailed)}")
    
    # Calculate GPR F1 scores
    print("\nCalculating GPR F1 scores...")
    df_gpr_f1 = calculate_f1_scores(df_gpr_detailed, method_name="GPR")
    
    # ========== Combine Results ==========
    print("\n" + "=" * 80)
    print("COMBINING RESULTS")
    print("=" * 80)
    
    # Combine FANS and GPR F1 scores into one dataframe
    df_combined = pd.concat([df_fans_f1, df_gpr_f1], ignore_index=True)
    
    # Save combined F1 scores
    combined_csv = str(ANALYSIS_DIR / "f1_scores_comparison.csv")
    df_combined.to_csv(combined_csv, index=False)
    print(f"\nCombined F1 scores saved to: {combined_csv}")
    
    # Display results
    print("\nCombined F1 Scores (FANS vs GPR):")
    print(df_combined.to_string(index=False))
    
    # ========== Summary Statistics ==========
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nFANS Performance:")
    print(f"  Average Accuracy: {df_fans_f1['accuracy'].mean():.4f}")
    print(f"  Average F1 Macro: {df_fans_f1['f1_macro'].mean():.4f}")
    print(f"  Average F1 Function: {df_fans_f1['f1_function'].mean():.4f}")
    print(f"  Average F1 Noise: {df_fans_f1['f1_noise'].mean():.4f}")
    print(f"  Total nodes classified: {df_fans_f1['total_classified'].sum()}")
    print(f"  Total correct: {df_fans_f1['correct_classified'].sum()}")
    
    print("\nGPR Performance:")
    print(f"  Average Accuracy: {df_gpr_f1['accuracy'].mean():.4f}")
    print(f"  Average F1 Macro: {df_gpr_f1['f1_macro'].mean():.4f}")
    print(f"  Average F1 Function: {df_gpr_f1['f1_function'].mean():.4f}")
    print(f"  Average F1 Noise: {df_gpr_f1['f1_noise'].mean():.4f}")
    print(f"  Total nodes classified: {df_gpr_f1['total_classified'].sum()}")
    print(f"  Total correct: {df_gpr_f1['correct_classified'].sum()}")
    
    print("\nPerformance Difference (FANS - GPR):")
    print(f"  Accuracy difference: {df_fans_f1['accuracy'].mean() - df_gpr_f1['accuracy'].mean():.4f}")
    print(f"  F1 Macro difference: {df_fans_f1['f1_macro'].mean() - df_gpr_f1['f1_macro'].mean():.4f}")
    print(f"  F1 Function difference: {df_fans_f1['f1_function'].mean() - df_gpr_f1['f1_function'].mean():.4f}")
    print(f"  F1 Noise difference: {df_fans_f1['f1_noise'].mean() - df_gpr_f1['f1_noise'].mean():.4f}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()