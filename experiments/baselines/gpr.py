#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, norm
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from math import lgamma
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import gc

# Set random seeds for reproducibility
np.random.seed(42)

# Helper function to calculate Gaussian Kernel matrix
def gaussian_kernel(X, length_scale=1.0):
    """Computes the Gaussian (RBF) kernel matrix efficiently."""
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) - 2 * np.dot(X, X.T) + np.sum(X**2, axis=1)
    K = np.exp(-sq_dists / (2 * length_scale**2))
    return K

# Optimized EM algorithm for VCM I - CPU only for stability
def run_em_vcm1_optimized(Y, G1, max_iter=100, tol=1e-4, verbose=False):
    """
    Optimized EM algorithm - stays on CPU for better memory management.
    """
    n = len(Y)
    Y = Y.reshape(-1, 1)  # Ensure Y is column vector
    
    # Initialize parameters
    tau2_sq_t = np.var(Y)
    if tau2_sq_t < 1e-6:
        tau2_sq_t = 1e-6
    tau1_sq_t = tau2_sq_t * 0.1
    
    # Precompute eigen-decomposition with better stability
    jitter = 1e-6 * np.trace(G1) / n
    try:
        eigenvalues, eigenvectors = eigh(G1 + np.identity(n) * jitter)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        U = eigenvectors.T @ Y
    except np.linalg.LinAlgError:
        return None, None, -np.inf
    
    for i in range(max_iter):
        tau1_sq_t = max(tau1_sq_t, 1e-8)
        tau2_sq_t = max(tau2_sq_t, 1e-8)
        
        D_diag = tau1_sq_t * eigenvalues + tau2_sq_t
        if np.any(D_diag <= 0):
            return None, None, -np.inf
            
        # Update parameters
        tau1_sq_new = (1 / n) * np.sum((eigenvalues / D_diag**2) * (U.flatten()**2))
        tau2_sq_new = (1 / n) * np.sum((1 / D_diag**2) * (U.flatten()**2))
        
        # Check convergence
        delta = max(abs(tau1_sq_new - tau1_sq_t), abs(tau2_sq_new - tau2_sq_t))
        if delta < tol:
            tau1_sq_t, tau2_sq_t = tau1_sq_new, tau2_sq_new
            break
        tau1_sq_t, tau2_sq_t = tau1_sq_new, tau2_sq_new
    
    # Calculate final log-likelihood
    D_diag = tau1_sq_t * eigenvalues + tau2_sq_t
    log_det_Sigma = np.sum(np.log(D_diag))
    Y_SigmaInv_Y = np.sum((U.flatten()**2) / D_diag)
    log_likelihood = -0.5 * (log_det_Sigma + Y_SigmaInv_Y + n * np.log(2 * np.pi))
    
    return float(tau1_sq_t), float(tau2_sq_t), float(log_likelihood)

def fit_h1_model_mle(Y_scaled, K_n, Z):
    """
    Find MLE for H1 hypothesis (group-specific functions, common noise).
    Parameters: [delta_1^2, delta_2^2, ..., sigma_0^2]
    """
    unique_groups = np.unique(Z)
    num_groups = len(unique_groups)
    n = len(Y_scaled)

    # Check group sizes
    for h in unique_groups:
        if np.sum(Z == h) < 3:  # Minimum 3 data points required
            return -np.inf, None

    # Pre-compute data and kernels for each group
    group_data = {}
    for h in unique_groups:
        indices = np.where(Z == h)[0]
        K_h = K_n[np.ix_(indices, indices)]
        
        # Apply stronger jitter
        jitter_h = max(1e-6 * np.mean(np.diag(K_h)), 1e-8)
        
        # Pre-compute eigen-decomposition
        try:
            eigvals, eigvecs = eigh(K_h + np.identity(len(indices)) * jitter_h)
            idx = np.argsort(eigvals)[::-1]
            
            # Safer eigenvalue handling
            eigvals_sorted = eigvals[idx]
            eigvals_sorted = np.maximum(eigvals_sorted, 1e-12)
            
            # Check condition number
            cond_num = eigvals_sorted[0] / eigvals_sorted[-1]
            if cond_num > 1e12:  # Too ill-conditioned
                return -np.inf, None
            
            group_data[h] = {
                "Y_h": Y_scaled[indices],
                "eigvals": eigvals_sorted,
                "eigvecs": eigvecs[:, idx]
            }
        except np.linalg.LinAlgError:
            return -np.inf, None

    # Objective function for optimization (Negative Log-Likelihood)
    def neg_log_likelihood(params):
        try:
            # params: [delta_1_sq, delta_2_sq, ..., common_sigma_sq]
            delta_sqs = params[:-1]
            sigma_sq = params[-1]

            # Stricter boundary conditions
            if sigma_sq <= 1e-10 or sigma_sq > 1e6: 
                return np.inf
            
            total_log_likelihood = 0
            
            for i, h in enumerate(unique_groups):
                delta_sq_h = delta_sqs[i]
                if delta_sq_h <= 1e-10 or delta_sq_h > 1e6: 
                    return np.inf

                data = group_data[h]
                Y_h, eigvals_h, eigvecs_h = data["Y_h"], data["eigvals"], data["eigvecs"]
                U_h = eigvecs_h.T @ Y_h
                
                D_diag = delta_sq_h * eigvals_h + sigma_sq
                
                # Safer checks
                if np.any(D_diag <= 1e-12) or np.any(~np.isfinite(D_diag)):
                    return np.inf

                log_det_Sigma_h = np.sum(np.log(D_diag))
                Y_SigmaInv_Y_h = np.sum((U_h.flatten()**2) / D_diag)
                
                # Check for NaN/Inf
                if not np.isfinite(log_det_Sigma_h) or not np.isfinite(Y_SigmaInv_Y_h):
                    return np.inf
                
                log_likelihood_h = -0.5 * (log_det_Sigma_h + Y_SigmaInv_Y_h + len(Y_h) * np.log(2 * np.pi))
                
                if not np.isfinite(log_likelihood_h):
                    return np.inf
                    
                total_log_likelihood += log_likelihood_h
                
            if not np.isfinite(total_log_likelihood):
                return np.inf
                
            return -total_log_likelihood
            
        except (ValueError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
            return np.inf

    # More robust initial value setting
    initial_params = []
    initial_sigma_sqs = []
    
    for h in unique_groups:
        data = group_data[h]
        try:
            d_sq, s_sq, _ = run_em_vcm1_optimized(data["Y_h"], K_n[np.ix_(np.where(Z==h)[0], np.where(Z==h)[0])])
            if d_sq is not None and s_sq is not None and np.isfinite(d_sq) and np.isfinite(s_sq):
                initial_params.append(max(d_sq, 1e-6))
                initial_sigma_sqs.append(max(s_sq, 1e-6))
            else:
                # Fallback initial values
                y_var = np.var(data["Y_h"])
                initial_params.append(max(y_var * 0.5, 1e-6))
                initial_sigma_sqs.append(max(y_var * 0.5, 1e-6))
        except:
            # Extreme fallback
            y_var = np.var(data["Y_h"]) if len(data["Y_h"]) > 1 else 1.0
            initial_params.append(max(y_var * 0.5, 1e-6))
            initial_sigma_sqs.append(max(y_var * 0.5, 1e-6))
    
    # Common sigma initial value
    if len(initial_sigma_sqs) > 0:
        initial_params.append(max(np.mean(initial_sigma_sqs), 1e-6))
    else:
        initial_params.append(1e-3)
    
    # More conservative bounds
    bounds = [(1e-8, 1e4)] * (num_groups + 1)
    
    # Try multiple optimization methods
    optimization_methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    for method in optimization_methods:
        try:
            result = minimize(neg_log_likelihood, initial_params, 
                            method=method, bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            
            if result.success and np.isfinite(result.fun):
                max_log_likelihood = -result.fun
                estimated_params = result.x
                
                # Validate results
                if np.all(np.isfinite(estimated_params)) and np.all(estimated_params > 0):
                    return max_log_likelihood, estimated_params
                    
        except Exception:
            continue
    
    # Return simple estimate if all methods fail
    try:
        # Combine results estimated independently for each group
        group_likelihoods = []
        for h in unique_groups:
            data = group_data[h]
            d_sq, s_sq, ll = run_em_vcm1_optimized(data["Y_h"], K_n[np.ix_(np.where(Z==h)[0], np.where(Z==h)[0])])
            if ll != -np.inf:
                group_likelihoods.append(ll)
        
        if len(group_likelihoods) == len(unique_groups):
            fallback_likelihood = sum(group_likelihoods)
            return fallback_likelihood, None
            
    except Exception:
        pass
    
    return -np.inf, None

def find_best_kernel_param_optimized(X, Y, initial_ls_range=(0.1, 10.0), verbose=False):
    """Optimized kernel parameter selection with memory management."""
    
    def objective(ls):
        if ls <= 1e-4:
            return np.inf
            
        K_n = gaussian_kernel(X, length_scale=ls)
        n = K_n.shape[0]
        jitter = 1e-6 * np.mean(np.diag(K_n))
        K_n_stable = K_n + np.identity(n) * jitter
        
        _, _, log_likelihood = run_em_vcm1_optimized(Y, K_n_stable, verbose=False)
        
        if log_likelihood == -np.inf:
            return np.inf
        return -log_likelihood

    result = minimize_scalar(objective, bounds=initial_ls_range, method='bounded')
    
    if result.success:
        best_ls = result.x
        max_log_likelihood = -result.fun
        return best_ls, max_log_likelihood
    else:
        fallback_ls = np.mean(initial_ls_range)
        _, _, max_log_likelihood = run_em_vcm1_optimized(Y, gaussian_kernel(X, fallback_ls), verbose=False)
        return fallback_ls, max_log_likelihood

# Function to calculate permutation size b_n
def calculate_bn_optimized(n, eigenvalues, xi_n_hat, alpha=0.05, alpha0_factor=1e-4, threshold_factor=1e-3):
    """Calculates the permutation size b_n based on estimated parameters."""
    alpha0 = alpha0_factor * alpha
    threshold = threshold_factor * alpha

    best_bn = 0
    for bn_candidate in range(1, n): # Check bn from 1 to n-1
        idx_cn = n - bn_candidate # Index for (n-bn+1)th largest eigenvalue (0-based)
        if idx_cn < 0: continue

        cn_plus_1 = eigenvalues[idx_cn] # (n-bn+1)th largest eigenvalue

        # Calculate omega_tilde_hat
        omega_tilde_hat = xi_n_hat * cn_plus_1

        # Calculate chi2 quantile Q_{b_n}(1-alpha0)
        # Need try-except for potential issues with chi2.ppf (e.g., df=0 or invalid alpha0)
        try:
            # Ensure bn_candidate > 0 for chi2
            if bn_candidate <= 0: continue
            q_bn = chi2.ppf(1 - alpha0, df=bn_candidate)
            if not np.isfinite(q_bn) or q_bn < 0: continue # Check for valid quantile
        except ValueError:
            continue # Skip if quantile calculation fails

        # Calculate v_tilde_hat
        term_inside_exp = 0.5 * omega_tilde_hat * q_bn
        # Avoid potential overflow in exp
        if term_inside_exp > 50: # Heuristic limit
            v_tilde_hat = np.inf
        else:
             try:
                 v_tilde_hat = 0.5 * np.exp(term_inside_exp) - 0.5
             except OverflowError:
                 v_tilde_hat = np.inf

        # Check condition
        if v_tilde_hat + alpha0 <= threshold:
            best_bn = bn_candidate
        else:
            # Since v_tilde increases with bn (usually), we can stop early
            break

    # Ensure bn is at least 1 if possible, otherwise 0.
    if best_bn == 0 and n > 1:
       # Check if bn=1 satisfies condition, maybe threshold was too strict?
       # Recalculate for bn=1 explicitly
        idx_cn1 = n - 1
        cn1 = eigenvalues[idx_cn1]
        omega_tilde_hat1 = xi_n_hat * cn1
        try:
             q_bn1 = chi2.ppf(1-alpha0, df=1)
             if np.isfinite(q_bn1) and q_bn1 >= 0:
                 term_inside_exp1 = 0.5 * omega_tilde_hat1 * q_bn1
                 if term_inside_exp1 <= 50:
                     v_tilde_hat1 = 0.5 * np.exp(term_inside_exp1) - 0.5
                     if v_tilde_hat1 + alpha0 <= threshold:
                          best_bn = 1
                 else: # Exp overflow case for bn=1
                      pass
             else: # Invalid quantile case for bn=1
                  pass
        except ValueError:
             pass # Quantile calc fails for bn=1

    return best_bn

def partial_permutation_test_optimized(env1_data, env2_data, y_node_col, dag_adjacency_matrix,
                                     num_permutations=100, alpha=0.05, gamma=0.1,
                                     length_scale_range=(0.1, 10.0),
                                     test_statistic_type='pseudo',
                                     verbose=True):
    """
    Memory-optimized version of the partial permutation test.
    """
    
    try:
        # --- 1. Data Preparation ---
        if verbose: print("1. Preparing Data...")
        y_node_idx = env1_data.columns.get_loc(y_node_col) if isinstance(y_node_col, str) else y_node_col
        
        parent_indices = np.where(dag_adjacency_matrix[:, y_node_idx] == 1)[0]
        
        if len(parent_indices) == 0:
            if verbose: print("No parents found, using intercept model")
            X1 = np.ones((len(env1_data), 1))
            X2 = np.ones((len(env2_data), 1))
        else:
            X1 = env1_data.iloc[:, parent_indices].values
            X2 = env2_data.iloc[:, parent_indices].values
            
        Y1 = env1_data.iloc[:, y_node_idx].values
        Y2 = env2_data.iloc[:, y_node_idx].values
        
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2)).reshape(-1, 1)
        Z = np.concatenate((np.ones(len(Y1)), 2 * np.ones(len(Y2)))).astype(int)
        n = len(Y)
        n1, n2 = len(Y1), len(Y2)
        
        # Memory check - if dataset is too large, reduce permutations
        if n > 5000:
            num_permutations = min(num_permutations, 50)
            if verbose: print(f"Large dataset detected (n={n}), reducing permutations to {num_permutations}")
        
        # Standardization
        x_scaler = StandardScaler().fit(X)
        X_scaled = x_scaler.transform(X)
        y_scaler = StandardScaler().fit(Y)
        Y_scaled = y_scaler.transform(Y)
        
        # --- 2. Kernel Parameter Selection ---
        if verbose: print("2. Selecting Kernel Parameter...")
        best_ls, h0_max_log_likelihood = find_best_kernel_param_optimized(
            X_scaled, Y_scaled, length_scale_range, verbose=verbose
        )
        
        if verbose: print(f"   Chosen length_scale = {best_ls:.4f}")
        
        K_n = gaussian_kernel(X_scaled, length_scale=best_ls)
        jitter_k = 1e-7 * np.mean(np.diag(K_n))
        K_n_stable = K_n + np.identity(n) * jitter_k
        
        # --- 3. Eigen-decomposition ---
        if verbose: print("3. Performing Eigen-decomposition...")
        try:
            eigenvalues, eigenvectors = eigh(K_n_stable)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            Gamma = eigenvectors
            C = eigenvalues
        except np.linalg.LinAlgError:
            return {"error": "Eigen-decomposition failed"}
        
        # --- 4. Parameter Estimation ---
        if verbose: print("4. Estimating Parameters...")
        delta0_sq_scaled_hat, sigma0_sq_hat, h0_max_ll_check = run_em_vcm1_optimized(
            Y_scaled, K_n_stable, verbose=False
        )
        
        if h0_max_ll_check == -np.inf:
            return {"error": "H0 MLE failed"}
            
        if sigma0_sq_hat < 1e-8:
            xi_n_hat = 1e10
        else:
            xi_n_hat = delta0_sq_scaled_hat / sigma0_sq_hat
        
        # --- 5. Calculate bn ---
        if verbose: print("5. Calculating bn...")
        bn = calculate_bn_optimized(n, C, xi_n_hat, alpha=alpha)
        if verbose: print(f"   Chosen bn = {bn}")
        
        if bn == 0:
            return {"p_value": 1.0, "observed_statistic": np.nan, "bn": 0, "message": "bn=0"}
        
        # --- 6. Observed Test Statistic ---
        if verbose: print("6. Calculating Observed Test Statistic...")
        
        # H_pseudo likelihood
        h1_max_log_likelihood, _ = fit_h1_model_mle(Y_scaled, K_n, Z)
        if h1_max_log_likelihood == -np.inf:
            return {"error": "H1 MLE failed for observed data"}

        observed_log_statistic = h1_max_log_likelihood - h0_max_log_likelihood

        # --- 7. Permutation Loop - Optimized ---
        if verbose: print(f"7. Running {num_permutations} Permutations (bn={bn})...")
        
        # Precompute projections
        W = Gamma.T @ Y_scaled
        perm_indices = np.arange(n - bn, n)
        
        permuted_log_statistics = []
        
        for i in range(num_permutations):
            if verbose and (i % 20 == 0):
                print(f"   Permutation {i+1}/{num_permutations}...")
            
            try:
                # Create permutation
                W_perm = W.copy()
                W_perm[perm_indices] = np.random.permutation(W_perm[perm_indices])
                Yp = Gamma @ W_perm
                
                # H0 likelihood for Yp
                _, _, h0_ll_p = run_em_vcm1_optimized(Yp, K_n_stable, verbose=False)
                if h0_ll_p == -np.inf:
                    continue
                
                # H1 likelihood for Yp (using new function)
                h1_ll_p, _ = fit_h1_model_mle(Yp, K_n, Z)
                if h1_ll_p == -np.inf:
                    continue
                    
                perm_stat = h1_ll_p - h0_ll_p
                permuted_log_statistics.append(perm_stat)
                    
            except Exception as e:
                if verbose: print(f"   Error in permutation {i+1}: {str(e)}")
                continue
            
            # Memory cleanup every 10 iterations
            if i % 10 == 0:
                gc.collect()
        
        # --- 8. Calculate p-value ---
        if verbose: print("8. Calculating p-value...")
        
        permuted_log_statistics = np.array(permuted_log_statistics)
        num_valid_perms = len(permuted_log_statistics)
        
        if num_valid_perms == 0:
            p_value = 1.0
        else:
            count_greater = np.sum(permuted_log_statistics >= observed_log_statistic)
            p_value = (count_greater + 1) / (num_valid_perms + 1)
        
        if verbose: print(f"   p-value = {p_value:.6f}")
        
        return {
            "p_value": p_value,
            "observed_log_statistic": float(observed_log_statistic),
            "bn": bn,
            "kernel_length_scale": float(best_ls),
            "h0_mle_delta0_sq_scaled": float(delta0_sq_scaled_hat),
            "h0_mle_sigma0_sq": float(sigma0_sq_hat),
            "h0_max_log_likelihood": float(h0_max_log_likelihood),
            "num_permutations_run": num_permutations,
            "num_valid_permutations": num_valid_perms
        }
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# --- Example Usage (Simulation Study in Original Paper of GPR(Li et al., 2021))---
if __name__ == '__main__':
    import argparse
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.gridspec import GridSpec

    # ----------------- CLI -----------------
    parser = argparse.ArgumentParser(description="GPR partial permutation simulation")
    parser.add_argument(
        "-f", "--functions",
        nargs="+",
        default=["all"],
        help="Function selection: List numbers like 1 3 6 or range 2-4, default all"
    )
    parser.add_argument(
        "--repetitions", "-r",
        type=int,
        default=30,
        help="Number of repetitions for each (case, function) combination (default 30)"
    )
    parser.add_argument(
        "--permutations", "-B",
        type=int,
        default=500,
        help="Number of permutations (default 500)"
    )
    args = parser.parse_args()

    np.random.seed(42)

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'

    # --- 6 functions from equation (28) ---
    def f0_case1(x1, x2): return (x1 + x2) / 2
    def f0_case2(x1, x2): return x1 * x2
    def f0_case3(x1, x2): return 2 * (x1 + x2)**3 / 15 - (x1 + x2) / 30
    def f0_case4(x1, x2): return 3 / (1 + x1**2 + x2**2) - 2
    def f0_case5(x1, x2): return np.sin(6 * x1) + x2
    def f0_case6(x1, x2): return np.sin(6 * x1 + 6 * x2)

    f0_functions = {
        1: ("f1: (x1+x2)/2", f0_case1),
        2: ("f2: x1*x2", f0_case2),
        3: ("f3: 2(x1+x2)^3/15 - (x1+x2)/30", f0_case3),
        4: ("f4: 3/(1+x1^2+x2^2) - 2", f0_case4),
        5: ("f5: sin(6*x1) + x2", f0_case5),
        6: ("f6: sin(6*x1 + 6*x2)", f0_case6),
    }

    # Parse selection
    def parse_selection(tokens):
        if any(t.lower() == "all" for t in tokens):
            return list(f0_functions.keys())
        selected = set()
        for t in tokens:
            if "-" in t:
                a, b = t.split("-", 1)
                if a.isdigit() and b.isdigit():
                    for k in range(int(a), int(b) + 1):
                        if k in f0_functions: selected.add(k)
            else:
                if t.isdigit() and int(t) in f0_functions:
                    selected.add(int(t))
        return sorted(selected)

    selected_func_ids = parse_selection(args.functions)
    if not selected_func_ids:
        print("No function selected. Exiting.")
        exit(0)

    # --- Table 1 cases ---
    case_configs = {
        "(a)": {"p1": 0.5, "a1": 0.5, "a2": 0.5},
        "(b)": {"p1": 0.2, "a1": 0.5, "a2": 0.5},
        "(c)": {"p1": 0.5, "a1": 0.8, "a2": 0.2},
        "(d)": {"p1": 0.2, "a1": 0.8, "a2": 0.2},
        "(e)": {"p1": 0.5, "a1": 1.0, "a2": 0.0}
    }

    # --- Data generation (equation 27) ---
    def generate_scenario2_data(n, p1, a1, a2, f0, sigma0=np.sqrt(0.1)):
        Z = np.random.choice([1, 2], size=n, p=[p1, 1 - p1])
        X = np.zeros((n, 2))
        for i in range(n):
            a_z = a1 if Z[i] == 1 else a2
            for j in range(2):
                if np.random.rand() < a_z:
                    X[i, j] = np.random.uniform(-1, 0)
                else:
                    X[i, j] = np.random.uniform(0, 1)
        Y = np.array([f0(x[0], x[1]) for x in X]) + np.random.normal(0, sigma0, size=n)
        return X, Y, Z

    n = 200
    num_repetitions = args.repetitions
    B = args.permutations

    simulation_results = {}

    # Total progress bar length
    total_tasks = len(selected_func_ids) * len(case_configs)
    task_counter = 0

    for func_id in selected_func_ids:
        f_label, f_func = f0_functions[func_id]
        for case_name, case_params in case_configs.items():
            task_counter += 1
            p_values = []
            loop_iter = tqdm(range(num_repetitions), desc=f"{f_label} {case_name}", leave=False)
            for _ in loop_iter:
                X, Y, Z = generate_scenario2_data(n=n, f0=f_func, **case_params)

                env1_mask = (Z == 1)
                env2_mask = (Z == 2)
                if np.sum(env1_mask) < 2 or np.sum(env2_mask) < 2:
                    continue

                env1_df = pd.DataFrame(X[env1_mask], columns=["X1", "X2"])
                env1_df["Y"] = Y[env1_mask]
                env2_df = pd.DataFrame(X[env2_mask], columns=["X1", "X2"])
                env2_df["Y"] = Y[env2_mask]

                dag_sim = np.zeros((3, 3))
                dag_sim[0, 2] = 1
                dag_sim[1, 2] = 1

                res = partial_permutation_test(
                    env1_data=env1_df,
                    env2_data=env2_df,
                    y_node_col="Y",
                    dag_adjacency_matrix=dag_sim,
                    num_permutations=B,
                    verbose=False
                )
                p_values.append(res["p_value"])
            simulation_results[(case_name, f_label)] = p_values

    print("Simulations complete. Plotting results...")

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    function_expressions = {
        "f1: (x1+x2)/2": r'$x_1+x_2$',
        "f2: x1*x2": r'$x_1x_2$',
        "f3: 2(x1+x2)^3/15 - (x_1+x_2)/30": r'$(x_1+x_2)^3 - \frac{(x_1+x_2)}{4}$',
        "f4: 3/(1+x1^2+x2^2) - 2": r'$(1+x_1^2+x_2^2)^{-1}$',
        "f5: sin(6*x1) + x2": r'$\sin(6x_1)+x_2$',
        "f6: sin(6*x1 + 6*x2)": r'$\sin(6x_1+6x_2)$'
    }

    case_styles = {
        "(a)": {"color": "black", "linestyle": "-", "marker": "o"},
        "(b)": {"color": "red", "linestyle": "--", "marker": "s"},
        "(c)": {"color": "green", "linestyle": "-.", "marker": "^"},
        "(d)": {"color": "blue", "linestyle": ":", "marker": "D"},
        "(e)": {"color": "cyan", "linestyle": "-", "marker": "p"}
    }

    # Only selected function labels in order
    selected_labels = [f0_functions[i][0] for i in selected_func_ids]

    plot_idx = 0
    for f_label in selected_labels:
        row, col = divmod(plot_idx, 3)
        ax = fig.add_subplot(gs[row, col])

        for case_name in case_configs.keys():
            p_vals = np.array(simulation_results.get((case_name, f_label), []))
            if len(p_vals) == 0:
                continue
            sorted_p = np.sort(p_vals)
            ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
            style = case_styles[case_name]
            ax.step(sorted_p, ecdf,
                    label=f"Case {case_name}",
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Uniform (Ref.)')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.2)); ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel('p-value'); ax.set_ylabel('empirical CDF')
        ax.set_title(function_expressions.get(f_label, f_label), fontsize=12)
        plot_idx += 1

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 0.98), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_name = 'figure2_reproduction.png' if len(selected_labels) == 6 else 'figure2_subset.png'
    plt.savefig(out_name, dpi=300)
    print(f"Figure saved as {out_name}")