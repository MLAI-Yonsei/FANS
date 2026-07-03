import os
import json
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Set, Tuple, Optional
import argparse
import glob
from pathlib import Path
from sklearn.metrics import f1_score

# Project root directory
SCRIPT_DIR = Path(__file__).parent
FANS_ROOT = SCRIPT_DIR.parent.parent  
DATA_DIR = "/data1/statduck/"
# FANS_ROOT / "data"
RESULTS_DIR = FANS_ROOT / "experiments" / "results"

def calculate_f1_score(true_set: Set[int], pred_set: Set[int]) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score
    
    Returns:
        (precision, recall, f1)
    """
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1

def get_nodes_with_parents(adj_matrix: np.ndarray) -> Set[int]:
    """
    Get nodes that have at least one parent (in-degree > 0)
    
    Args:
        adj_matrix: Adjacency matrix where adj[parent, child]
    
    Returns:
        Set of node indices with parents
    """
    nodes_with_parents = set()
    if adj_matrix is not None and adj_matrix.ndim == 2:
        for node_idx in range(adj_matrix.shape[1]):
            if np.sum(adj_matrix[:, node_idx]) > 0:
                nodes_with_parents.add(int(node_idx))
    return nodes_with_parents

def load_ground_truth(data_dir: str, node_count: int, graph_type: str, dataset_idx: int) -> Optional[Dict]:
    """
    Load ground truth from adjacency matrix and any available result file
    
    Returns:
        Dictionary with:
        - adj_matrix
        - nodes_with_parents
    """
    # Load adjacency matrix
    adj_file = os.path.join(data_dir, f"nodes_{node_count}", graph_type, f"adj_{dataset_idx}.npy")
    if not os.path.exists(adj_file):
        return None
    
    adj_matrix = np.load(adj_file)
    nodes_with_parents = get_nodes_with_parents(adj_matrix)
    
    return {
        'adj_matrix': adj_matrix,
        'nodes_with_parents': nodes_with_parents
    }

def load_dataset_info_from_any_result(results_dirs: Dict[str, str], node_count: int, graph_type: str, dataset_idx: int) -> Optional[Dict]:
    """
    Load dataset_info from any available result file (try all methods)
    """
    # Try each method's result file
    for method_name, results_dir in results_dirs.items():
        if method_name == "fans":
            file_pattern = os.path.join(results_dir, f"nodes_{node_count}", graph_type, f"fans_*.json")
        elif method_name == "gpr":
            file_pattern = os.path.join(results_dir, f"nodes_{node_count}", graph_type, f"gpr_*.json")
        elif method_name == "iscan":
            file_pattern = os.path.join(results_dir, f"nodes_{node_count}", graph_type, f"iscan_*.json")
        elif method_name == "linearccp":
            file_pattern = os.path.join(results_dir, f"nodes_{node_count}", graph_type, f"linearccp_*.json")
        elif method_name == "splitkci":
            file_pattern = os.path.join(results_dir, f"nodes_{node_count}", graph_type, f"splitkci_nodes{node_count}_{graph_type}_{dataset_idx}.json")
        elif method_name == "prediter":
            file_pattern = os.path.join(results_dir, f"nodes_{node_count}", graph_type, f"prediter_nodes{node_count}_{graph_type}_{dataset_idx}.json")
        else:
            continue
        
        # Try to find and load the file
        if "*" in file_pattern:
            files = glob.glob(file_pattern)
            matching_files = []
            for f in files:
                basename = os.path.basename(f)
                
                # Special handling for GPR files with format: gpr_optimized_nodes10_ER_9_cpu.json
                if method_name == "gpr" and basename.startswith('gpr_'):
                    match = re.search(rf'_{graph_type}_(\d+)', basename)
                    if match and int(match.group(1)) == dataset_idx:
                        matching_files.append(f)
                else:
                    # Default pattern for other methods: _(\d+).json$
                    match = re.search(r'_(\d+)\.json$', f)
                    if match and int(match.group(1)) == dataset_idx:
                        matching_files.append(f)
                        
            if matching_files:
                file_path = matching_files[0]
            else:
                continue
        else:
            file_path = file_pattern
            if not os.path.exists(file_path):
                continue
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        dataset_info = data.get("dataset_info", {})
        if dataset_info:
            return dataset_info
    
    return None

def load_fans_results(results_dir: str, node_count: int, graph_type: str, dataset_idx: int, nodes_with_parents: Set[int]) -> Dict:
    """
    Load FANS results
    """
    result = {
        'shifted_nodes': set(),
        'function_shifts': set(),
        'noise_shifts': set(),
        'error': None
    }
    
    # Updated pattern to match the nested structure:
    file_pattern = os.path.join(results_dir, f"*nodes_{node_count}_{graph_type}_adj_{dataset_idx}", "*", "fans_analysis", "fans_*.json")
    files = glob.glob(file_pattern)
    
    if not files:
        result['error'] = f"File not found with pattern: {file_pattern}"
        return result
    
    # Use the first matching file (there should typically be only one per dataset)
    target_file = files[0]
    
    if not os.path.exists(target_file):
        result['error'] = "File not found"
        return result
    
    with open(target_file, 'r') as f:
        data = json.load(f)
    
    # Check for the new structure (direct keys) or old structure (nested under "fans")
    if "comparison_results" in data and "independence_results" in data:
        # New structure: direct access
        comparison_results = data.get("comparison_results", {})
        independence_results = data.get("independence_results", {})
    elif "fans" in data:
        # Old structure: nested under "fans"
        fans_data = data["fans"]
        
        if "error" in fans_data:
            result['error'] = fans_data['error']
            return result
        
        comparison_results = fans_data.get("comparison_results", {})
        independence_results = fans_data.get("independence_results", {})
    else:
        result['error'] = "No fans results"
        return result
    
    # Get detected shifted nodes from comparison_results
    detected_shifted = set(comparison_results.get("shifted_nodes_threshold", []))
    result['shifted_nodes'] = detected_shifted.intersection(nodes_with_parents)
    
    # Get function/noise classification from estimated_shift_types_elbow (new structure)
    estimated_shift_types = independence_results.get("estimated_shift_types", {})
    
    fans_function_shifts = set()
    fans_noise_shifts = set()
    
    for node_id_str, shift_type in estimated_shift_types.items():
        node_int = int(node_id_str)
        if shift_type == "function":
            fans_function_shifts.add(node_int)
        elif shift_type == "noise":
            fans_noise_shifts.add(node_int)
    
    result['function_shifts'] = fans_function_shifts
    result['noise_shifts'] = fans_noise_shifts
    return result

def load_unified_results(results_dir: str, node_count: int, graph_type: str, dataset_idx: int,
                         nodes_with_parents: Set[int], method: str, true_shifted_nodes: Set[int] = None) -> Dict:
    """
    Unified loader for all methods (except FANS which has special structure).
    
    All methods now use unified naming:
    - shifted_nodes: detected shifted nodes
    - estimated_shift_types: {"node_id": "function" or "noise"}
    
    Args:
        results_dir: Directory containing results
        node_count: Number of nodes
        graph_type: Graph type (ER/SF)
        dataset_idx: Dataset index
        nodes_with_parents: Set of nodes with parents
        method: Method name (iscan, linearccp, splitkci, prediter, gpr)
        true_shifted_nodes: Ground truth shifted nodes (for filtering)
    """
    result = {
        'shifted_nodes': set(),
        'function_shifts': set(),
        'noise_shifts': set(),
        'error': None
    }
    
    # Method-specific file path patterns
    file_patterns = {
        'iscan': f"iscan_*_{dataset_idx}.json",
        'linearccp': f"linearccp_*_{dataset_idx}.json",
        'splitkci': f"splitkci_nodes{node_count}_{graph_type}_{dataset_idx}.json",
        'prediter': f"prediter_nodes{node_count}_{graph_type}_{dataset_idx}.json",
        'gpr': f"result_dataset_{dataset_idx}.json",
        'lcit': f"lcit_nodes{node_count}_{graph_type}_{dataset_idx}.json"
    }
    
    # Data key in JSON (None means direct access)
    data_keys = {
        'iscan': 'iscan',
        'linearccp': 'linearccp', 
        'splitkci': 'splitkci',
        'prediter': None,
        'gpr': None,
        'lcit': 'lcit'
    }
    
    # Find file
    dir_path = os.path.join(results_dir, f"nodes_{node_count}", graph_type)
    pattern = file_patterns.get(method)
    
    if '*' in pattern:
        files = glob.glob(os.path.join(dir_path, pattern))
        file_path = files[0] if files else None
    else:
        file_path = os.path.join(dir_path, pattern)
        if not os.path.exists(file_path):
            file_path = None
    
    if not file_path:
        result['error'] = "File not found"
        return result
    
    # Load JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        result['error'] = f"Error loading JSON: {str(e)}"
        return result
    
    # Extract method-specific data
    data_key = data_keys.get(method)
    if data_key:
        if data_key not in data:
            result['error'] = f"No {method} results"
            return result
        method_data = data[data_key]
        if "error" in method_data:
            result['error'] = method_data['error']
            return result
    else:
        method_data = data
    
    # Get shifted_nodes (unified naming, with fallback for old names)
    shifted_nodes = set(method_data.get("shifted_nodes", 
                        method_data.get("detected_shifted_nodes",
                        method_data.get("detected_shifts", []))))
    result['shifted_nodes'] = shifted_nodes.intersection(nodes_with_parents)
    
    # Get estimated_shift_types (unified naming)
    estimated_shift_types = method_data.get("estimated_shift_types", {})
    
    if estimated_shift_types:
        pred_function = {int(n) for n, t in estimated_shift_types.items() if t == "function"}
        pred_noise = {int(n) for n, t in estimated_shift_types.items() if t == "noise"}
        
        if true_shifted_nodes:
            result['function_shifts'] = pred_function.intersection(true_shifted_nodes).intersection(nodes_with_parents)
            result['noise_shifts'] = pred_noise.intersection(true_shifted_nodes).intersection(nodes_with_parents)
        else:
            result['function_shifts'] = pred_function.intersection(nodes_with_parents)
            result['noise_shifts'] = pred_noise.intersection(nodes_with_parents)
    
    return result

def parse_shift_types(shift_types: Dict, shifted_nodes: Set[int]) -> Tuple[Set[int], Set[int]]:
    """
    Parse shift types from dataset_info
    
    Returns:
        (function_shifts, noise_shifts)
    """
    function_shifts = set()
    noise_shifts = set()
    
    for node in shifted_nodes:
        shift_value = shift_types.get(str(node), "unknown")
        
        # Handle both list and string formats
        if isinstance(shift_value, list):
            shift_value = shift_value[0] if len(shift_value) > 0 else "unknown"
        
        # Check if it's a function shift
        if shift_value in ["function", "sin_cos", "pa_delete"]:
            function_shifts.add(node)
        # Check if it's a noise shift
        elif shift_value.startswith("noise") or shift_value in ["noise", "noise_scale", "noise_variance"]:
            noise_shifts.add(node)
    
    return function_shifts, noise_shifts

def analyze_all_methods(results_dirs: Dict[str, str], data_dir: str, output_csv: str = "unified_analysis_results.csv"):
    """
    Analyze all methods and create a unified dataframe
    
    Args:
        results_dirs: Dictionary mapping method names to their results directories
        data_dir: Directory containing adjacency matrices
        output_csv: Output CSV filename
    """
    
    node_counts = [10, 20, 30, 40, 50]
    graph_types = ["ER", "SF"]
    dataset_indices = list(range(1, 31))
    
    rows = []
    
    for node_count in node_counts:
        for graph_type in graph_types:
            for dataset_idx in dataset_indices:
                print(f"Processing: nodes={node_count}, graph={graph_type}, dataset={dataset_idx}")
                
                row = {
                    'node_count': node_count,
                    'graph_type': graph_type,
                    'dataset_idx': dataset_idx
                }
                
                # Load ground truth (adjacency matrix)
                gt = load_ground_truth(data_dir, node_count, graph_type, dataset_idx)
                if gt is None:
                    print(f"  Warning: Could not load ground truth, skipping")
                    continue
                
                nodes_with_parents = gt['nodes_with_parents']
                
                # Load dataset_info from any available result file
                dataset_info = load_dataset_info_from_any_result(results_dirs, node_count, graph_type, dataset_idx)
                if dataset_info is None:
                    print(f"  Warning: Could not load dataset_info, skipping")
                    continue
                
                # Parse true shifted nodes and shift types
                true_shifted_original = set(dataset_info.get("shifted_nodes", []))
                true_shifted = true_shifted_original.intersection(nodes_with_parents)
                
                shift_types = dataset_info.get("shift_types", {})
                true_function_shifts, true_noise_shifts = parse_shift_types(shift_types, true_shifted)
                
                # Store ground truth
                row['true_shifted_nodes'] = sorted(list(true_shifted))
                row['true_function_shifts'] = sorted(list(true_function_shifts))
                row['true_noise_shifts'] = sorted(list(true_noise_shifts))
                row['nodes_with_parents'] = sorted(list(nodes_with_parents))
                
                # Load FANS results (special structure)
                for fans_key in ['fans', 'fans_swap', 'fans_5000', 'fans_10000', 'fans_30000', 'fans_100epochs', 'fans_500epochs', 'fans_dim8', 'fans_dim16', 'fans_anm', 'fans_small']:
                    if fans_key not in results_dirs:
                        continue
                    method_result = load_fans_results(results_dirs[fans_key], node_count, graph_type, dataset_idx, nodes_with_parents)
                    row[f'{fans_key}_shifted_nodes'] = sorted(list(method_result['shifted_nodes']))
                    row[f'{fans_key}_function_shifts'] = sorted(list(method_result['function_shifts']))
                    row[f'{fans_key}_noise_shifts'] = sorted(list(method_result['noise_shifts']))
                    row[f'{fans_key}_error'] = method_result['error']
                
                # Load all other methods using unified loader
                unified_methods = ['iscan', 'linearccp', 'splitkci', 'prediter', 'gpr', 'lcit']
                for method_name in unified_methods:
                    if method_name not in results_dirs:
                        continue
                    
                    method_result = load_unified_results(
                        results_dirs[method_name], node_count, graph_type, dataset_idx,
                        nodes_with_parents, method_name, true_shifted
                    )
                    
                    row[f'{method_name}_shifted_nodes'] = sorted(list(method_result['shifted_nodes']))
                    row[f'{method_name}_function_shifts'] = sorted(list(method_result['function_shifts']))
                    row[f'{method_name}_noise_shifts'] = sorted(list(method_result['noise_shifts']))
                    row[f'{method_name}_error'] = method_result['error']
                
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Calculate F1 scores for each method
    print("\nCalculating F1 scores...")
    calculate_f1_scores(df)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    print(f"Total rows: {len(df)}")
    
    return df

def verify_classification_completeness(df: pd.DataFrame):
    """
    Verify that all ground truth shifted nodes are properly classified as function or noise by each method
    
    Returns:
        Dictionary with verification results for each method
    """
    methods = ['fans', 'fans_swap', 'fans_5000', 'fans_10000', 'fans_30000', 'fans_100epochs', 'fans_500epochs', 'fans_dim8', 'fans_dim16', 'fans_anm', 'fans_small', 'gpr', 'iscan', 'linearccp']
    results = {}
    
    for method in methods:
        shifted_col = f'{method}_shifted_nodes'
        function_col = f'{method}_function_shifts'
        noise_col = f'{method}_noise_shifts'
        
        if shifted_col not in df.columns:
            continue
        
        total_datasets = 0
        mismatches = 0
        
        for idx, row in df.iterrows():
            # Get ground truth shifted nodes
            true_shifted = set(row['true_shifted_nodes']) if isinstance(row['true_shifted_nodes'], list) else set()
            
            # Get method's function/noise classification
            pred_function = set(row[function_col]) if isinstance(row[function_col], list) else set()
            pred_noise = set(row[noise_col]) if isinstance(row[noise_col], list) else set()
            if len(true_shifted) == 0:
                continue
            
            total_datasets += 1
            
            # Check if all true_shifted nodes are classified as function or noise
            pred_classified = pred_function.union(pred_noise)
            true_shifted_classified = true_shifted.intersection(pred_classified)
                
            if true_shifted != true_shifted_classified:
                mismatches += 1
        
        results[method] = {
            'total_datasets': total_datasets,
            'mismatches': mismatches,
            'completeness_rate': (total_datasets - mismatches) / max(total_datasets, 1)
        }
    
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETENESS VERIFICATION")
    print("(Checking if all true shifted nodes are classified as function or noise)")
    print("="*80)
    for method, stats in results.items():
        print(f"{method.upper():12s}: {stats['total_datasets'] - stats['mismatches']}/{stats['total_datasets']} complete ({stats['completeness_rate']*100:.1f}%)")
    
    return results

def calculate_f1_scores(df: pd.DataFrame):
    """
    Calculate F1 scores for all methods and add them to the dataframe
    """
    methods = ['fans', 'fans_swap', 'fans_5000', 'fans_10000', 'fans_30000', 'fans_100epochs', 'fans_500epochs', 'fans_dim8', 'fans_dim16', 'fans_anm', 'fans_small', 'gpr', 'iscan', 'linearccp', 'splitkci', 'prediter', 'lcit']
    
    for method in methods:
        shifted_col = f'{method}_shifted_nodes'
        function_col = f'{method}_function_shifts'
        noise_col = f'{method}_noise_shifts'
        
        if shifted_col not in df.columns:
            continue
        
        # F1 for shifted node detection
        f1_shifted_list = []
        f1_function_list = []
        f1_noise_list = []
        f1_macro_list = []
        
        for idx, row in df.iterrows():
            true_shifted = set(row['true_shifted_nodes']) if isinstance(row['true_shifted_nodes'], list) else set()
            pred_shifted = set(row[shifted_col]) if isinstance(row[shifted_col], list) else set()
            
            _, _, f1_shifted = calculate_f1_score(true_shifted, pred_shifted)
            f1_shifted_list.append(f1_shifted)
            
            # F1 for function/noise classification (only if method supports it)
            if method not in ['splitkci', 'prediter', 'lcit']:
                true_function = set(row['true_function_shifts']) if isinstance(row['true_function_shifts'], list) else set()
                pred_function = set(row[function_col]) if isinstance(row[function_col], list) else set()
                
                true_noise = set(row['true_noise_shifts']) if isinstance(row['true_noise_shifts'], list) else set()
                pred_noise = set(row[noise_col]) if isinstance(row[noise_col], list) else set()
                true_classified = true_function.union(true_noise)
                if true_shifted != true_classified:
                    print(f"\n⚠️  WARNING [{method.upper()}] Dataset {idx}: Ground truth mismatch!")
                    print(f"    True shifted: {sorted(true_shifted)}")
                    print(f"    True classified (func+noise): {sorted(true_classified)}")
                    print(f"    Missing: {sorted(true_shifted - true_classified)}")

                tp_function = 0
                fn_function = 0
                tp_noise = 0
                fn_noise = 0
                
                for node in true_shifted:
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
                
                f1_function_list.append(f1_function)
                f1_noise_list.append(f1_noise)
                f1_macro_list.append(f1_macro)
            else:
                f1_function_list.append(np.nan)
                f1_noise_list.append(np.nan)
                f1_macro_list.append(np.nan)
        
        df[f'{method}_f1_shifted'] = f1_shifted_list
        df[f'{method}_f1_function'] = f1_function_list
        df[f'{method}_f1_noise'] = f1_noise_list
        
        if method not in ['splitkci', 'prediter', 'lcit']:
            df[f'{method}_f1_macro'] = f1_macro_list

def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics for each method
    """
    methods = ['fans', 'fans_swap', 'fans_5000', 'fans_10000', 'fans_30000', 'fans_100epochs', 'fans_500epochs', 'fans_dim8', 'fans_dim16', 'fans_anm', 'fans_small', 'gpr', 'iscan', 'linearccp', 'splitkci', 'prediter', 'lcit']
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Average F1 Scores ± Std Dev)")
    print("="*80)
    
    for graph_type in ["ER", "SF"]:
        print(f"\n{'='*80}")
        print(f"GRAPH TYPE: {graph_type}")
        print(f"{'='*80}")
        
        for node_count in [10, 20, 30, 40, 50]:
            subset = df[(df['node_count'] == node_count) & (df['graph_type'] == graph_type)]
            
            if len(subset) == 0:
                continue
            
            print(f"\n  Nodes: {node_count} (Total: {len(subset)} datasets)")
            print("  " + "-" * 76)
            
            for method in methods:
                f1_shifted_col = f'{method}_f1_shifted'
                f1_macro_col = f'{method}_f1_macro'
                error_col = f'{method}_error'
                
                if f1_shifted_col not in df.columns:
                    print(f"    {method.upper():12s}: No results")
                    continue
                
                # fans_swap only has 15 datasets; average over loaded datasets
                # (error is None) so missing runs don't drag the mean down with F1=0.
                if method == 'fans_swap' and error_col in df.columns:
                    method_subset = subset[subset[error_col].isna()]
                else:
                    method_subset = subset
                
                if len(method_subset) == 0:
                    print(f"    {method.upper():12s}: No loaded datasets")
                    continue
                
                # Count successful loads (error is None)
                if error_col in df.columns:
                    successful_count = method_subset[error_col].isna().sum() + (method_subset[error_col] == None).sum()
                else:
                    successful_count = len(method_subset)
                
                avg_f1_shifted = method_subset[f1_shifted_col].mean()
                std_f1_shifted = method_subset[f1_shifted_col].std()
                
                if method not in ['splitkci', 'prediter', 'lcit'] and f1_macro_col in df.columns:
                    avg_f1_macro = method_subset[f1_macro_col].mean()
                    std_f1_macro = method_subset[f1_macro_col].std()
                    print(f"    {method.upper():12s}: F1_Shifted={avg_f1_shifted:.3f}±{std_f1_shifted:.3f}, F1_Macro={avg_f1_macro:.3f}±{std_f1_macro:.3f} ({successful_count}/{len(method_subset)} loaded)")
                else:
                    print(f"    {method.upper():12s}: F1_Shifted={avg_f1_shifted:.3f}±{std_f1_shifted:.3f} ({successful_count}/{len(method_subset)} loaded)")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL DATA LOADING SUMMARY")
    print("="*80)
    
    for method in methods:
        error_col = f'{method}_error'
        if error_col in df.columns:
            successful_count = df[error_col].isna().sum() + (df[error_col] == None).sum()
            total_count = len(df)
            print(f"{method.upper():12s}: {successful_count}/{total_count} datasets loaded successfully ({100*successful_count/total_count:.1f}%)")
        else:
            print(f"{method.upper():12s}: No results found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified analysis of all causal discovery methods')
    parser.add_argument('--fans_dir', type=str, default=str("/data1/statduck/exp1_result_fans"), 
                       help='Directory containing FANS results')
    parser.add_argument('--fans_swap_dir', type=str, default=str("/data1/statduck/exp1_result_fans_swap"), 
                       help='Directory containing FANS swap results')
    parser.add_argument('--fans_5000_dir', type=str, default=str("/data1/statduck/exp1_result_fans_5000"), 
                       help='Directory containing FANS 5000 sample results')
    parser.add_argument('--fans_10000_dir', type=str, default=str("/data1/statduck/exp1_result_fans_10000"), 
                       help='Directory containing FANS 10000 sample results')
    parser.add_argument('--fans_30000_dir', type=str, default=str("/data1/statduck/exp1_result_fans_30000"), 
                       help='Directory containing FANS 30000 sample results')
    parser.add_argument('--fans_100epochs_dir', type=str, default=str("/data1/statduck/exp1_result_fans_100epochs"), 
                       help='Directory containing FANS 100 epochs results')
    parser.add_argument('--fans_500epochs_dir', type=str, default=str("/data1/statduck/exp1_result_fans_500epochs"), 
                       help='Directory containing FANS 500 epochs results')
    parser.add_argument('--fans_dim8_dir', type=str, default=str("/data1/statduck/exp1_result_fans_dim8"), 
                       help='Directory containing FANS dim8 results')
    parser.add_argument('--fans_dim16_dir', type=str, default=str("/data1/statduck/exp1_result_fans_dim16"), 
                       help='Directory containing FANS dim16 results')
    parser.add_argument('--fans_anm_dir', type=str, default=str("/data1/statduck/exp1_result_fans_anm"), 
                       help='Directory containing FANS ANM results')
    parser.add_argument('--fans_small_dir', type=str, default=str("/data1/statduck/exp1_result_fans_small"), 
                       help='Directory containing FANS small (num_samples=1000) results')
    parser.add_argument('--gpr_dir', type=str, default=str("/data1/statduck/exp1_result_gpr"), 
                       help='Directory containing GPR results')
    parser.add_argument('--iscan_dir', type=str, default=str(RESULTS_DIR / "iscan"), 
                       help='Directory containing ISCAN results')
    parser.add_argument('--linearccp_dir', type=str, default=str(RESULTS_DIR / "linearccp_n=50000"), 
                       help='Directory containing LinearCCP results')
    parser.add_argument('--splitkci_dir', type=str, default=str(RESULTS_DIR / "splitkci"), 
                       help='Directory containing SplitKCI results')
    parser.add_argument('--prediter_dir', type=str, default=str(RESULTS_DIR / "prediter_paramtuning"), 
                       help='Directory containing PreDITEr results')
    parser.add_argument('--lcit_dir', type=str, default=str(RESULTS_DIR / "lcit_n=1000"), 
                       help='Directory containing LCIT results')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR + "data"), 
                       help='Directory containing adjacency matrices')
    parser.add_argument('--output_csv', type=str, default='unified_analysis_results.csv', 
                       help='Output CSV filename')
    args = parser.parse_args()
    
    # Create results directories mapping
    results_dirs = {
        'fans': args.fans_dir,
        'fans_swap': args.fans_swap_dir,
        'fans_5000': args.fans_5000_dir,
        'fans_10000': args.fans_10000_dir,
        'fans_30000': args.fans_30000_dir,
        'fans_100epochs': args.fans_100epochs_dir,
        'fans_500epochs': args.fans_500epochs_dir,
        'fans_dim8': args.fans_dim8_dir,
        'fans_dim16': args.fans_dim16_dir,
        'fans_anm': args.fans_anm_dir,
        'fans_small': args.fans_small_dir,
        'gpr': args.gpr_dir,
        'iscan': args.iscan_dir,
        'linearccp': args.linearccp_dir,
        'splitkci': args.splitkci_dir,
        'prediter': args.prediter_dir,
        'lcit': args.lcit_dir
    }
    
    # Run analysis
    df = analyze_all_methods(results_dirs, args.data_dir, args.output_csv)
    
    # Verify classification completeness
    verify_classification_completeness(df)
    
    # Print summary
    print_summary_statistics(df)