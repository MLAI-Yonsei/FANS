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
        - ...nodes_10_ER_lambda_1.00_adj_1 -> (10, 'ER', 1)
    """
    pattern_violation = r'nodes_(\d+)_(ER|SF)_lambda_[\d.]+_adj_(\d+)'
    match = re.search(pattern_violation, dir_name)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3))

    pattern1 = r'nodes_(\d+)_(ER|SF)_adj_(\d+)'
    match = re.search(pattern1, dir_name)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3))

    raise ValueError(f"Could not parse directory name: {dir_name}")

def load_metadata(nodes: int, graph_type: str, adj_num: int, violation_lambda: str = None) -> Dict:
    if violation_lambda:
        metadata_path = Path(f"/data1/statduck/data_both/data_violation/lambda_{violation_lambda}/nodes_{nodes}/{graph_type}/metadata_{adj_num}.json")
    else:
        metadata_path = DATA_DIR / "data_both" / "data" / f"nodes_{nodes}" / graph_type / f"metadata_{adj_num}.json"
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_fans_results(fans_json_path: str) -> Dict:
    with open(fans_json_path, 'r') as f:
        return json.load(f)

def has_fans_results(fans_analysis_dir: Path) -> bool:
    """Check if fans_results.json exists and has valid data."""
    if not fans_analysis_dir.exists() or not fans_analysis_dir.is_dir():
        return False
    
    fans_json = fans_analysis_dir / 'fans_results.json'
    if not fans_json.exists():
        return False
    
    # Check if independence_results has estimated_shift_types
    with open(fans_json, 'r') as f:
        data = json.load(f)
    
    independence_results = data.get('independence_results', {})
    return 'estimated_shift_types' in independence_results

def extract_shift_info(metadata: Dict) -> Tuple[List[int], List[int], List[int]]:
    shifted_nodes = metadata.get('shifted_nodes', [])
    shift_types = metadata.get('shift_types', {})
    true_function_only = []
    true_function_and_noise = []
    
    for node_idx in shifted_nodes:
        node_key = str(node_idx)
        if node_key in shift_types:
            shift_list = shift_types[node_key]
            has_function = 'function_shift' in shift_list
            has_noise = ('variance_shift' in shift_list) or ('distribution_shift' in shift_list)
            if has_function and not has_noise:
                true_function_only.append(node_idx)
            elif has_function and has_noise:
                true_function_and_noise.append(node_idx)
    return shifted_nodes, true_function_only, true_function_and_noise

def extract_detection_info(fans_results: Dict) -> Tuple[List[int], List[int], List[int]]:
    analyzed_nodes = fans_results.get('analyzed_nodes', [])
    
    detected_function_only = []
    detected_function_and_noise = []
    
    # First try simultaneous_shift_results, then fall back to independence_results
    simultaneous_shift_results = fans_results.get('simultaneous_shift_results', None)
    
    if simultaneous_shift_results is not None:
        for node_idx in analyzed_nodes:
            node_key = str(node_idx)
            if node_key in simultaneous_shift_results:
                shift_type = simultaneous_shift_results[node_key].get('shift_type', None)
                
                if shift_type == 'function_only':
                    detected_function_only.append(node_idx)
                elif shift_type == 'function_and_noise':
                    detected_function_and_noise.append(node_idx)
    else:
        # Use independence_results.estimated_shift_types
        independence_results = fans_results.get('independence_results', {})
        estimated_shift_types = independence_results.get('estimated_shift_types', {})
        
        for node_idx in analyzed_nodes:
            node_key = str(node_idx)
            if node_key in estimated_shift_types:
                shift_type = estimated_shift_types[node_key]
                
                # Map shift types: "function" -> function_only, "simultaneous" -> function_and_noise
                if shift_type == 'function':
                    detected_function_only.append(node_idx)
                elif shift_type in ['simultaneous', 'noise', 'function_and_noise']:
                    detected_function_and_noise.append(node_idx)
    
    return analyzed_nodes, detected_function_only, detected_function_and_noise


def find_experiment_folders(output_testing_dir: str) -> List[Tuple[str, str]]:
    results = []
    output_path = Path(output_testing_dir)
    
    for parent_dir in output_path.iterdir():
        if not parent_dir.is_dir():
            continue
    
        nodes, graph_type, adj_num = parse_directory_name(parent_dir.name)
        
        found_valid_exp = False
        for exp_dir in parent_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            fans_analysis_dir = exp_dir / 'fans_analysis'
            fans_json = fans_analysis_dir / 'fans_results.json'
            
            if fans_json.exists() and has_fans_results(fans_analysis_dir):
                results.append((parent_dir.name, str(fans_json)))
                print(f"Selected {exp_dir.name} for {parent_dir.name}")
                found_valid_exp = True
                break
        
        if not found_valid_exp:
            print(f"No valid experiment found for {parent_dir.name} (missing fans_results.json or estimated_shift_types)")
    return results

def create_results_dataframe(output_testing_dir: str, violation_lambda: str = None) -> pd.DataFrame:
    experiment_folders = find_experiment_folders(output_testing_dir)
    
    print(f"\nFound {len(experiment_folders)} experiment folders with simultaneous shift analysis")
    
    rows = []
    
    for parent_dir_name, fans_json_path in experiment_folders:
        nodes, graph_type, adj_num = parse_directory_name(parent_dir_name)
        metadata = load_metadata(nodes, graph_type, adj_num, violation_lambda=violation_lambda)
        fans_results = load_fans_results(fans_json_path)
        shifted_nodes, true_function_only, true_function_and_noise = extract_shift_info(metadata)
        analyzed_nodes, detected_function_only, detected_function_and_noise = extract_detection_info(fans_results)

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

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['nodes', 'graph_type', 'adj_num']).reset_index(drop=True)
    
    return df


def calculate_f1_scores(df: pd.DataFrame, method_name: str = "FANS") -> pd.DataFrame:
    """Compute per-dataset F1 (function / noise / macro) then macro-average within
    each (nodes, graph_type) group. Mirrors the aggregation strategy used in
    experiments/analysis/analysis.py.
    """
    if len(df) == 0:
        print(f"Warning: Empty dataframe for {method_name}, returning empty F1 scores")
        return pd.DataFrame(columns=['method', 'nodes', 'graph_type', 'num_experiments',
                                     'total_classified', 'correct_classified', 'accuracy',
                                     'f1_function', 'f1_noise', 'f1_macro'])

    per_dataset_records = []

    for _, row in df.iterrows():
        analyzed_shifted = [n for n in row['analyzed_nodes'] if n in row['shifted_nodes']]

        true_function = set(row['true_function_only'])
        true_noise = set(row['true_function_and_noise'])
        pred_function = set(row['detected_function_only'])
        pred_noise = set(row['detected_function_and_noise'])

        tp_function = 0
        fn_function = 0
        tp_noise = 0
        fn_noise = 0

        for node in analyzed_shifted:
            if node in true_function:
                if node in pred_function:
                    tp_function += 1
                else:
                    fn_function += 1
            elif node in true_noise:
                if node in pred_noise:
                    tp_noise += 1
                else:
                    fn_noise += 1

        fp_function = fn_noise
        fp_noise = fn_function

        if tp_function + fp_function + fn_function > 0:
            precision_func = tp_function / max(tp_function + fp_function, 1)
            recall_func = tp_function / max(tp_function + fn_function, 1)
            if precision_func + recall_func > 0:
                f1_function = 2 * (precision_func * recall_func) / (precision_func + recall_func)
            else:
                f1_function = 0.0
        else:
            f1_function = 0.0

        if tp_noise + fp_noise + fn_noise > 0:
            precision_noise = tp_noise / max(tp_noise + fp_noise, 1)
            recall_noise = tp_noise / max(tp_noise + fn_noise, 1)
            if precision_noise + recall_noise > 0:
                f1_noise = 2 * (precision_noise * recall_noise) / (precision_noise + recall_noise)
            else:
                f1_noise = 0.0
        else:
            f1_noise = 0.0

        if (tp_function + fn_function > 0) and (tp_noise + fn_noise > 0):
            f1_macro = (f1_function + f1_noise) / 2.0
        elif tp_function + fn_function > 0:
            f1_macro = f1_function
        elif tp_noise + fn_noise > 0:
            f1_macro = f1_noise
        else:
            f1_macro = 0.0

        total = tp_function + fn_function + tp_noise + fn_noise
        correct = tp_function + tp_noise
        accuracy = correct / max(total, 1)

        per_dataset_records.append({
            'nodes': row['nodes'],
            'graph_type': row['graph_type'],
            'f1_function': f1_function,
            'f1_noise': f1_noise,
            'f1_macro': f1_macro,
            'accuracy': accuracy,
            'total_classified': total,
            'correct_classified': correct,
        })

    df_per_dataset = pd.DataFrame(per_dataset_records)

    results = []
    for (nodes, graph_type), group in df_per_dataset.groupby(['nodes', 'graph_type']):
        results.append({
            'method': method_name,
            'nodes': nodes,
            'graph_type': graph_type,
            'num_experiments': len(group),
            'total_classified': int(group['total_classified'].sum()),
            'correct_classified': int(group['correct_classified'].sum()),
            'accuracy': group['accuracy'].mean(),
            'f1_function': group['f1_function'].mean(),
            'f1_noise': group['f1_noise'].mean(),
            'f1_macro': group['f1_macro'].mean(),
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
        json_file = base_path / f"result_dataset_{idx}.json"
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
        gpr_result = load_gpr_result(gpr_json_path)
        
        # Extract dataset info
        dataset_info = gpr_result.get('dataset_info', {})
        node_count = dataset_info.get('node_count', nodes)
        g_type = dataset_info.get('graph_type', graph_type)
        adj_num = dataset_info.get('dataset_index', -1)
        
        # Load original metadata file for ground truth (instead of using dataset_info)
        metadata = load_metadata(node_count, g_type, adj_num)
        
        # Extract shift information from original metadata (ground truth)
        shifted_nodes, true_function_only, true_function_and_noise = extract_shift_info(metadata)
        
        # Extract detection information (predictions)
        analyzed_nodes, detected_function_only, detected_function_and_noise = extract_gpr_detection_info(gpr_result)
        
        # Create row
        row = {
            'nodes': node_count,
            'graph_type': g_type,
            'adj_num': adj_num,
            'shifted_nodes': shifted_nodes,
            'analyzed_nodes': analyzed_nodes,
            'true_function_only': true_function_only,
            'true_function_and_noise': true_function_and_noise,
            'detected_function_only': detected_function_only,
            'detected_function_and_noise': detected_function_and_noise,
        }
        
        rows.append(row)
            
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

def print_method_summary(label: str, df_f1: pd.DataFrame):
    print(f"\n{label} Performance:")
    print(f"  Average Accuracy: {df_f1['accuracy'].mean():.4f}")
    print(f"  Average F1 Macro: {df_f1['f1_macro'].mean():.4f}")
    print(f"  Average F1 Function: {df_f1['f1_function'].mean():.4f}")
    print(f"  Average F1 Noise: {df_f1['f1_noise'].mean():.4f}")
    print(f"  Total nodes classified: {df_f1['total_classified'].sum()}")
    print(f"  Total correct: {df_f1['correct_classified'].sum()}")


def main():
    """Main execution function."""
    output_testing_dir = str(RESULTS_DIR / "both" / "exp2_fans")
    violation_dirs = {
        "0.50": str(RESULTS_DIR / "both" / "exp2_fans_violation_lambda_0.50"),
        "1.00": str(RESULTS_DIR / "both" / "exp2_fans_violation_lambda_1.00"),
    }
    gpr_dir = str(RESULTS_DIR / "both" / "exp2_gpr")
    
    print("=" * 80)
    print("FANS and GPR Results Analysis - Function Only vs Function + Noise")
    print("=" * 80)
    
    # ========== FANS Analysis ==========
    print("\n" + "=" * 80)
    print("ANALYZING FANS RESULTS")
    print("=" * 80)
    
    print("\nCreating FANS detailed results dataframe...")
    df_fans_detailed = create_results_dataframe(output_testing_dir)
    
    print(f"\nTotal FANS experiments processed: {len(df_fans_detailed)}")
    print(f"\nBreakdown by nodes and graph type:")
    print(df_fans_detailed.groupby(['nodes', 'graph_type']).size())
    
    print("\nCalculating FANS F1 scores...")
    df_fans_f1 = calculate_f1_scores(df_fans_detailed, method_name="FANS")
    
    # ========== FANS Violation Analysis (lambda = 0.50, 1.00) ==========
    df_violation_f1_dict = {}
    for lam, violation_dir in violation_dirs.items():
        print("\n" + "=" * 80)
        print(f"ANALYZING FANS VIOLATION (lambda={lam}) RESULTS")
        print("=" * 80)
        
        print(f"\nCreating FANS violation (lambda={lam}) detailed results dataframe...")
        df_violation_detailed = create_results_dataframe(violation_dir, violation_lambda=lam)
        
        print(f"\nTotal FANS violation (lambda={lam}) experiments processed: {len(df_violation_detailed)}")
        if len(df_violation_detailed) > 0:
            print(f"\nBreakdown by nodes and graph type:")
            print(df_violation_detailed.groupby(['nodes', 'graph_type']).size())
        
        print(f"\nCalculating FANS violation (lambda={lam}) F1 scores...")
        df_violation_f1_dict[lam] = calculate_f1_scores(
            df_violation_detailed, method_name=f"FANS_violation_lambda_{lam}"
        )
    
    # ========== GPR Analysis ==========
    print("\n" + "=" * 80)
    print("ANALYZING GPR RESULTS")
    print("=" * 80)
    
    print("\nCreating GPR detailed results dataframe...")
    df_gpr_detailed = create_gpr_results_dataframe(
        gpr_dir=gpr_dir,
        nodes=10,
        graph_type='ER',
        start_idx=1,
        end_idx=30
    )
    
    print(f"\nTotal GPR experiments processed: {len(df_gpr_detailed)}")
    
    print("\nCalculating GPR F1 scores...")
    df_gpr_f1 = calculate_f1_scores(df_gpr_detailed, method_name="GPR")
    
    # ========== Combine Results ==========
    print("\n" + "=" * 80)
    print("COMBINING RESULTS")
    print("=" * 80)
    
    df_combined = pd.concat(
        [df_fans_f1] + [df_violation_f1_dict[lam] for lam in violation_dirs] + [df_gpr_f1],
        ignore_index=True,
    )
    
    combined_csv = str(ANALYSIS_DIR / "f1_scores_comparison.csv")
    df_combined.to_csv(combined_csv, index=False)
    print(f"\nCombined F1 scores saved to: {combined_csv}")
    
    print("\nCombined F1 Scores (FANS vs FANS_violation vs GPR):")
    print(df_combined.to_string(index=False))
    
    # ========== Summary Statistics ==========
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print_method_summary("FANS", df_fans_f1)
    for lam in violation_dirs:
        print_method_summary(f"FANS_violation (lambda={lam})", df_violation_f1_dict[lam])
    print_method_summary("GPR", df_gpr_f1)
    
    print("\nPerformance Difference (FANS - GPR):")
    print(f"  Accuracy difference: {df_fans_f1['accuracy'].mean() - df_gpr_f1['accuracy'].mean():.4f}")
    print(f"  F1 Macro difference: {df_fans_f1['f1_macro'].mean() - df_gpr_f1['f1_macro'].mean():.4f}")
    
    for lam, df_violation_f1 in df_violation_f1_dict.items():
        if len(df_violation_f1) > 0:
            print(f"\nPerformance Difference (FANS_violation lambda={lam} - GPR):")
            print(f"  Accuracy difference: {df_violation_f1['accuracy'].mean() - df_gpr_f1['accuracy'].mean():.4f}")
            print(f"  F1 Macro difference: {df_violation_f1['f1_macro'].mean() - df_gpr_f1['f1_macro'].mean():.4f}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()