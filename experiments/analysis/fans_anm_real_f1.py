"""
Evaluate fans_anm against its CORRECT ground truth (data_anm/), not data/.

This is a one-off diagnostic: analysis.py uses /data1/statduck/data/ as the
ground truth source, but fans_anm was trained on /data1/statduck/data_anm/,
which has different DAGs and shifted_nodes for nodes=50 (nodes=10 happens to
match between the two). We re-score fans_anm with the matching ground truth.
"""
import glob
import json
import os
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

DATA_ANM_DIR = "/data1/statduck/data_anm"
FANS_ANM_DIR = "/data1/statduck/exp1_result_fans_anm"


def calculate_f1(true_set: Set[int], pred_set: Set[int]) -> Tuple[float, float, float]:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def get_nodes_with_parents(adj: np.ndarray) -> Set[int]:
    return {int(i) for i in range(adj.shape[1]) if np.sum(adj[:, i]) > 0}


def parse_shift_types(shift_types: Dict, shifted_nodes: Set[int]) -> Tuple[Set[int], Set[int]]:
    function_shifts, noise_shifts = set(), set()
    for node in shifted_nodes:
        v = shift_types.get(str(node), "unknown")
        if isinstance(v, list):
            v = v[0] if v else "unknown"
        if v in ["function", "sin_cos", "pa_delete"]:
            function_shifts.add(node)
        elif v.startswith("noise") or v in ["noise", "noise_scale", "noise_variance"]:
            noise_shifts.add(node)
    return function_shifts, noise_shifts


def load_fans_anm_prediction(
    node_count: int, graph_type: str, idx: int, nodes_with_parents: Set[int]
) -> Dict:
    pattern = os.path.join(
        FANS_ANM_DIR,
        f"*nodes_{node_count}_{graph_type}_adj_{idx}",
        "*",
        "fans_analysis",
        "fans_results.json",
    )
    files = glob.glob(pattern)
    if not files:
        return {"error": f"no result for {pattern}"}
    with open(files[0]) as f:
        data = json.load(f)
    cmp_res = data.get("comparison_results", {})
    indep_res = data.get("independence_results", {})
    pred_shifted = set(cmp_res.get("shifted_nodes_threshold", [])) & nodes_with_parents
    est = indep_res.get("estimated_shift_types", {})
    pred_function = {int(n) for n, t in est.items() if t == "function"} & nodes_with_parents
    pred_noise = {int(n) for n, t in est.items() if t == "noise"} & nodes_with_parents
    return {
        "shifted": pred_shifted,
        "function": pred_function,
        "noise": pred_noise,
    }


def evaluate(node_count: int, graph_type: str = "ER", n_datasets: int = 30) -> pd.DataFrame:
    rows = []
    for idx in range(1, n_datasets + 1):
        adj_path = os.path.join(DATA_ANM_DIR, f"nodes_{node_count}", graph_type, f"adj_{idx}.npy")
        meta_path = os.path.join(DATA_ANM_DIR, f"nodes_{node_count}", graph_type, f"metadata_{idx}.json")
        if not os.path.exists(adj_path) or not os.path.exists(meta_path):
            continue
        adj = np.load(adj_path)
        with open(meta_path) as f:
            meta = json.load(f)

        nodes_with_parents = get_nodes_with_parents(adj)
        true_shifted = set(meta.get("shifted_nodes", [])) & nodes_with_parents
        true_function, true_noise = parse_shift_types(meta.get("shift_types", {}), true_shifted)

        pred = load_fans_anm_prediction(node_count, graph_type, idx, nodes_with_parents)
        if "error" in pred:
            continue

        _, _, f1_shifted = calculate_f1(true_shifted, pred["shifted"])

        # function/noise classification F1 (same logic as analysis.py)
        tp_f = fn_f = tp_n = fn_n = 0
        for node in true_shifted:
            if node in true_function:
                if node in pred["function"]:
                    tp_f += 1
                else:
                    fn_f += 1
            elif node in true_noise:
                if node in pred["noise"]:
                    tp_n += 1
                else:
                    fn_n += 1
        fp_f, fp_n = fn_n, fn_f

        def f1_from_counts(tp, fp, fn):
            if tp + fp + fn == 0:
                return 0.0
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            return 0.0 if p + r == 0 else 2 * p * r / (p + r)

        f1_function = f1_from_counts(tp_f, fp_f, fn_f)
        f1_noise = f1_from_counts(tp_n, fp_n, fn_n)
        if (tp_f + fn_f > 0) and (tp_n + fn_n > 0):
            f1_macro = (f1_function + f1_noise) / 2
        elif tp_f + fn_f > 0:
            f1_macro = f1_function
        elif tp_n + fn_n > 0:
            f1_macro = f1_noise
        else:
            f1_macro = 0.0

        rows.append(
            {
                "idx": idx,
                "n_true_shifted": len(true_shifted),
                "n_pred_shifted": len(pred["shifted"]),
                "f1_shifted": f1_shifted,
                "f1_function": f1_function,
                "f1_noise": f1_noise,
                "f1_macro": f1_macro,
            }
        )
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("FANS_ANM real F1 (evaluated against data_anm/ ground truth)")
    print("=" * 70)
    for node_count in [10, 50]:
        df = evaluate(node_count, "ER", n_datasets=30)
        if df.empty:
            print(f"\nnodes_{node_count}/ER: no data")
            continue
        print(f"\nnodes_{node_count}/ER  (n={len(df)} datasets)")
        print("-" * 70)
        print(f"  F1_shifted : {df['f1_shifted'].mean():.4f} ± {df['f1_shifted'].std():.4f}")
        print(f"  F1_function: {df['f1_function'].mean():.4f} ± {df['f1_function'].std():.4f}")
        print(f"  F1_noise   : {df['f1_noise'].mean():.4f} ± {df['f1_noise'].std():.4f}")
        print(f"  F1_macro   : {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}")
        print(f"  Per-dataset F1_macro:")
        for _, r in df.iterrows():
            print(
                f"    idx={int(r['idx']):2d}  n_true={int(r['n_true_shifted']):2d}  "
                f"n_pred={int(r['n_pred_shifted']):2d}  "
                f"F1_sh={r['f1_shifted']:.3f}  F1_macro={r['f1_macro']:.3f}"
            )


if __name__ == "__main__":
    main()
