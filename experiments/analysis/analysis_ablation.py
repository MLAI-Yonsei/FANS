import os
import json
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


DATA_DIR = "/data1/statduck/data"
FANS_RESULT_DIR = "/data1/statduck/exp1_result_fans"

NODE_COUNT = 50
GRAPH_TYPE = "ER"
DATASET_INDICES = list(range(1, 31))


def get_nodes_with_parents(adj_matrix: np.ndarray) -> Set[int]:
    nodes_with_parents = set()
    if adj_matrix is not None and adj_matrix.ndim == 2:
        for node_idx in range(adj_matrix.shape[1]):
            if np.sum(adj_matrix[:, node_idx]) > 0:
                nodes_with_parents.add(int(node_idx))
    return nodes_with_parents


def calculate_f1(true_set: Set[int], pred_set: Set[int]) -> float:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_shift_types(shift_types: Dict, shifted_nodes: Set[int]) -> Tuple[Set[int], Set[int]]:
    function_shifts = set()
    noise_shifts = set()
    for node in shifted_nodes:
        shift_value = shift_types.get(str(node), "unknown")
        if isinstance(shift_value, list):
            shift_value = shift_value[0] if shift_value else "unknown"
        if shift_value in ["function", "sin_cos", "pa_delete"]:
            function_shifts.add(node)
        elif shift_value.startswith("noise") or shift_value in ["noise", "noise_scale", "noise_variance"]:
            noise_shifts.add(node)
    return function_shifts, noise_shifts


def load_all_datasets() -> List[Dict]:
    """Load ground truth and raw FANS results for all 30 datasets (ER, nodes=10)."""
    datasets = []

    for idx in DATASET_INDICES:
        adj_file = os.path.join(DATA_DIR, f"nodes_{NODE_COUNT}", GRAPH_TYPE, f"adj_{idx}.npy")
        meta_file = os.path.join(DATA_DIR, f"nodes_{NODE_COUNT}", GRAPH_TYPE, f"metadata_{idx}.json")

        if not os.path.exists(adj_file) or not os.path.exists(meta_file):
            print(f"  Skipping dataset {idx}: missing adj or metadata")
            continue

        adj_matrix = np.load(adj_file)
        nodes_with_parents = get_nodes_with_parents(adj_matrix)

        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        true_shifted = set(metadata.get("shifted_nodes", [])).intersection(nodes_with_parents)
        shift_types = metadata.get("shift_types", {})
        true_function, true_noise = parse_shift_types(shift_types, true_shifted)

        file_pattern = os.path.join(
            FANS_RESULT_DIR,
            f"*nodes_{NODE_COUNT}_{GRAPH_TYPE}_adj_{idx}",
            "*", "fans_analysis", "fans_*.json"
        )
        files = glob.glob(file_pattern)
        if not files:
            print(f"  Skipping dataset {idx}: no FANS result found")
            continue

        with open(files[0], 'r') as f:
            fans_data = json.load(f)

        if "comparison_results" in fans_data and "independence_results" in fans_data:
            comparison = fans_data["comparison_results"]
            independence = fans_data["independence_results"]
        elif "fans" in fans_data:
            comparison = fans_data["fans"].get("comparison_results", {})
            independence = fans_data["fans"].get("independence_results", {})
        else:
            print(f"  Skipping dataset {idx}: unrecognized FANS result format")
            continue

        # Per-node JS divergence (only for nodes with parents)
        node_js = {}
        for key, val in comparison.items():
            if isinstance(val, dict) and "node_idx" in val:
                nid = int(val["node_idx"])
                if nid in nodes_with_parents:
                    node_js[nid] = float(val["avg_js_divergence"])

        # Per-node dcor scores (only for originally detected shifted nodes)
        node_dcor_diff = {}
        for key, val in independence.items():
            if key == "estimated_shift_types":
                continue
            if isinstance(val, dict) and "env1_dcor_score" in val and "env2_dcor_score" in val:
                nid = int(val["node_idx"])
                node_dcor_diff[nid] = float(val["env2_dcor_score"]) - float(val["env1_dcor_score"])

        datasets.append({
            "idx": idx,
            "nodes_with_parents": nodes_with_parents,
            "true_shifted": true_shifted,
            "true_function": true_function,
            "true_noise": true_noise,
            "node_js": node_js,
            "node_dcor_diff": node_dcor_diff,
        })

    return datasets


def compute_classification_f1(
    true_function: Set[int], true_noise: Set[int],
    pred_function: Set[int], pred_noise: Set[int]
) -> Tuple[float, float, float]:
    """Compute per-class F1 and macro F1 for function/noise classification."""
    true_shifted = true_function | true_noise

    tp_f = sum(1 for n in true_shifted if n in true_function and n in pred_function)
    fn_f = sum(1 for n in true_shifted if n in true_function and n not in pred_function)
    tp_n = sum(1 for n in true_shifted if n in true_noise and n in pred_noise)
    fn_n = sum(1 for n in true_shifted if n in true_noise and n not in pred_noise)
    fp_f, fp_n = fn_n, fn_f

    def _f1(tp, fp, fn):
        if tp + fp + fn == 0:
            return 0.0
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    f1_f = _f1(tp_f, fp_f, fn_f)
    f1_n = _f1(tp_n, fp_n, fn_n)

    has_f = (tp_f + fn_f) > 0
    has_n = (tp_n + fn_n) > 0
    if has_f and has_n:
        macro = (f1_f + f1_n) / 2.0
    elif has_f:
        macro = f1_f
    elif has_n:
        macro = f1_n
    else:
        macro = 0.0

    return f1_f, f1_n, macro


def ablation_x(datasets: List[Dict], x_values: np.ndarray):
    """Sweep JS divergence threshold x → F1_shifted (detection)."""
    print("\n" + "=" * 80)
    print("ABLATION 1: JS Divergence Threshold (x) → F1_shifted")
    print(f"  Condition: node is shifted if avg_js_divergence > x")
    print(f"  {len(datasets)} datasets, nodes={NODE_COUNT}, graph={GRAPH_TYPE}")
    print("=" * 80)
    print(f"{'x':>8s}  {'F1_shifted':>12s}  {'Precision':>10s}  {'Recall':>10s}")
    print("-" * 46)

    results = []
    for x in x_values:
        f1_list, prec_list, rec_list = [], [], []
        for ds in datasets:
            pred_shifted = {nid for nid, js in ds["node_js"].items() if js > x}
            true_shifted = ds["true_shifted"]

            tp = len(true_shifted & pred_shifted)
            fp = len(pred_shifted - true_shifted)
            fn = len(true_shifted - pred_shifted)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            f1_list.append(f1)
            prec_list.append(p)
            rec_list.append(r)

        avg_f1 = np.mean(f1_list)
        avg_p = np.mean(prec_list)
        avg_r = np.mean(rec_list)
        print(f"{x:8.4f}  {avg_f1:12.4f}  {avg_p:10.4f}  {avg_r:10.4f}")
        results.append({"x": x, "f1_shifted": avg_f1, "precision": avg_p, "recall": avg_r})

    return results


def ablation_y(datasets: List[Dict], y_values: np.ndarray):
    """Sweep dcor-difference threshold y → F1_macro (classification).

    Uses the original FANS-detected shifted nodes (not re-thresholded).
    For nodes with dcor info: function if dcor_diff > y, else noise.
    """
    print("\n" + "=" * 80)
    print("ABLATION 2: dCor Difference Threshold (y) → F1_macro")
    print(f"  Condition: function shift if env2_dcor - env1_dcor > y, else noise")
    print(f"  {len(datasets)} datasets, nodes={NODE_COUNT}, graph={GRAPH_TYPE}")
    print("=" * 80)
    print(f"{'y':>8s}  {'F1_function':>12s}  {'F1_noise':>10s}  {'F1_macro':>10s}")
    print("-" * 46)

    results = []
    for y in y_values:
        f1_f_list, f1_n_list, f1_m_list = [], [], []
        for ds in datasets:
            detected_nodes = set(ds["node_dcor_diff"].keys())
            pred_function = {n for n in detected_nodes if ds["node_dcor_diff"][n] > y}
            pred_noise = detected_nodes - pred_function

            f1_f, f1_n, macro = compute_classification_f1(
                ds["true_function"], ds["true_noise"],
                pred_function, pred_noise
            )
            f1_f_list.append(f1_f)
            f1_n_list.append(f1_n)
            f1_m_list.append(macro)

        avg_f1_f = np.mean(f1_f_list)
        avg_f1_n = np.mean(f1_n_list)
        avg_macro = np.mean(f1_m_list)
        print(f"{y:8.4f}  {avg_f1_f:12.4f}  {avg_f1_n:10.4f}  {avg_macro:10.4f}")
        results.append({"y": y, "f1_function": avg_f1_f, "f1_noise": avg_f1_n, "f1_macro": avg_macro})

    return results


def ablation_xy(datasets: List[Dict], x_values: np.ndarray, y_values: np.ndarray):
    """Joint sweep of (x, y) → F1_shifted + F1_macro.

    x determines which nodes are detected as shifted (avg_js > x).
    y determines classification: function if dcor_diff > y, else noise.
    Only nodes that have dcor info AND pass the x threshold are classified.
    """
    print("\n" + "=" * 80)
    print("ABLATION 3: Joint (x, y) Sweep → F1_shifted & F1_macro")
    print(f"  {len(datasets)} datasets, nodes={NODE_COUNT}, graph={GRAPH_TYPE}")
    print("=" * 80)
    print(f"{'x':>8s}  {'y':>8s}  {'F1_shifted':>12s}  {'F1_function':>12s}  {'F1_noise':>10s}  {'F1_macro':>10s}")
    print("-" * 66)

    results = []
    for x in x_values:
        for y in y_values:
            f1_s_list, f1_f_list, f1_n_list, f1_m_list = [], [], [], []
            for ds in datasets:
                pred_shifted = {nid for nid, js in ds["node_js"].items() if js > x}
                f1_s = calculate_f1(ds["true_shifted"], pred_shifted)

                classifiable = pred_shifted & set(ds["node_dcor_diff"].keys())
                pred_function = {n for n in classifiable if ds["node_dcor_diff"][n] > y}
                pred_noise = classifiable - pred_function

                f1_f, f1_n, macro = compute_classification_f1(
                    ds["true_function"], ds["true_noise"],
                    pred_function, pred_noise
                )

                f1_s_list.append(f1_s)
                f1_f_list.append(f1_f)
                f1_n_list.append(f1_n)
                f1_m_list.append(macro)

            avg_s = np.mean(f1_s_list)
            avg_f = np.mean(f1_f_list)
            avg_n = np.mean(f1_n_list)
            avg_m = np.mean(f1_m_list)
            print(f"{x:8.4f}  {y:8.4f}  {avg_s:12.4f}  {avg_f:12.4f}  {avg_n:10.4f}  {avg_m:10.4f}")
            results.append({
                "x": x, "y": y,
                "f1_shifted": avg_s, "f1_function": avg_f,
                "f1_noise": avg_n, "f1_macro": avg_m
            })

    return results


if __name__ == "__main__":
    print("Loading datasets...")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} / {len(DATASET_INDICES)} datasets\n")

    # Show per-dataset JS divergence distribution for context
    print("=" * 80)
    print("Per-dataset JS divergence statistics (for reference)")
    print("=" * 80)
    all_js = []
    for ds in datasets:
        js_vals = list(ds["node_js"].values())
        all_js.extend(js_vals)
    all_js_arr = np.array(all_js)
    print(f"  Global: min={all_js_arr.min():.4f}, max={all_js_arr.max():.4f}, "
          f"mean={all_js_arr.mean():.4f}, median={np.median(all_js_arr):.4f}, "
          f"std={all_js_arr.std():.4f}")

    # Show per-dataset dcor diff distribution for context
    all_dcor = []
    for ds in datasets:
        all_dcor.extend(ds["node_dcor_diff"].values())
    all_dcor_arr = np.array(all_dcor)
    print(f"  dCor diff: min={all_dcor_arr.min():.4f}, max={all_dcor_arr.max():.4f}, "
          f"mean={all_dcor_arr.mean():.4f}, median={np.median(all_dcor_arr):.4f}, "
          f"std={all_dcor_arr.std():.4f}")

    x_values = np.arange(0.02, 0.32, 0.02)
    y_values = np.arange(-0.10, 0.52, 0.02)

    res_x = ablation_x(datasets, x_values)
    res_y = ablation_y(datasets, y_values)

    # Joint sweep with same 0.02 step
    res_xy = ablation_xy(datasets, x_values, y_values)
