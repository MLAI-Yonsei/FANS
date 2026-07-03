import os
import math
import torch
import numpy as np
from scipy import stats
from scipy.stats import chi2, kstest, gaussian_kde
from scipy.spatial.distance import jensenshannon
import dcor
from functools import reduce
import operator
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import GAM, s, te
from visualize import FANSVisualizer
from kneed import KneeLocator
import ipdb
import time

def find_elbow_shift_nodes(value_dict, online=True):
    if not value_dict or len(value_dict) == 0:
        print('No value_dict')
        return []
    df = pd.DataFrame()
    df.index = value_dict.keys()
    df['value'] = [x for x in value_dict.values()]
    df = df.sort_values('value', ascending=False)

    if df['value'].nunique() == 1:
        print('All values are the same')
        return []

    kn = KneeLocator(range(df.shape[0]), df['value'].values,
    curve='convex', direction='decreasing', online=online, interp_method='interp1d')
    print(kn.knee)
    shift_nodes = df.index[:kn.knee]
    if kn.knee is None: return []
    return shift_nodes.tolist()

class FANSAnalyzer:
    """
    FANS (Flow-based Analysis of Noise Shift) implementation
    Extracted from CausalNFightning for modular use
    """

    def __init__(self, model, preparator, device, input_scaler=None, external_dag_path=None,
                 enable_visualizer=True):
        """
        Initialize FANS analyzer
        
        Args:
            model: Trained causal normalizing flow model
            preparator: Data preparator with adjacency matrix and data generation methods
            device: PyTorch device (CPU/GPU)
            input_scaler: Input scaler
            external_dag_path: Path to external DAG
            enable_visualizer: If False, all plotting is skipped (analysis still runs).
        """
        self.model = model
        self.preparator = preparator
        self.device = device
        self.external_dag_path = external_dag_path
        self.input_scaler = input_scaler
        self.dag = self.preparator.adjacency(False).cpu().numpy()
        self.visualizer = FANSVisualizer() if enable_visualizer else None

    def get_x_norm(self, batch):
        """Get normalized x from batch (helper method)"""
        x_norm = self.input_scaler.transform(batch[0].to(self.device), inplace=False)
        x_norm = torch.where(torch.isnan(x_norm), torch.zeros_like(x_norm), x_norm)
        return x_norm
    
    def x_to_z(self, x):
        # Convert to batch format if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_norm = self.get_x_norm((x,))
        # Forward transform
        with torch.no_grad():
            z = self.model.flow().transform(x_norm)
        return z 
    
    def z_to_x(self, z):
        # Inverse transform
        with torch.no_grad():
            x_norm = self.model.flow().transform.inv(z)
        # Use input_scaler's inverse_transform
        x = self.input_scaler.inverse_transform(x_norm, inplace=False)
        return x 

    def decompose_probability(self, data, quantile_values=None):
        z_orig = self.x_to_z(data)
        n_samples, n_vars = data.shape
        quantiles = [0.25, 0.5, 0.75]
        
        if quantile_values is not None:
            if not isinstance(quantile_values, torch.Tensor):
                quantile_values = torch.tensor(quantile_values.astype(np.float32)).to(self.device)
            else:
                quantile_values = quantile_values.to(self.device)
        
        nodes_with_parents = []
        nodes_parents_map = {}
        
        for node_idx in range(n_vars):
            parents = np.where(self.dag[node_idx,:] == 1)[0]
            if len(parents) > 0:
                nodes_with_parents.append(node_idx)
                nodes_parents_map[node_idx] = parents
        n_nodes = len(nodes_with_parents)
        
        sampled_from_cond = {}
        chunk_size = (n_nodes + 1) // 6 if n_nodes > 400 else n_nodes
        n_chunks = (n_nodes + chunk_size - 1) // chunk_size

        print(f"Processing {n_nodes} nodes with {n_chunks} chunk(s) (chunk_size={chunk_size})")

        for q_idx, q_value in enumerate(quantiles):
            for chunk_idx in range(n_chunks):
                start_node_idx = chunk_idx * chunk_size
                end_node_idx = min((chunk_idx + 1) * chunk_size, n_nodes)
                current_nodes = nodes_with_parents[start_node_idx:end_node_idx]
                current_chunk_size = len(current_nodes)
                
                print(f"  Quantile {q_idx+1}/{len(quantiles)}, Chunk {chunk_idx+1}/{n_chunks}: nodes {start_node_idx} to {end_node_idx-1}")
                
                batch_data = data.repeat(current_chunk_size, 1)
                batch_z_orig = z_orig.repeat(current_chunk_size, 1)

                for batch_idx, node_idx in enumerate(current_nodes):
                    start = batch_idx * n_samples
                    end = (batch_idx + 1) * n_samples
                    parents = nodes_parents_map[node_idx]

                    for parent_idx in parents:
                        batch_data[start:end, parent_idx] = quantile_values[q_idx, parent_idx]
                
                batch_fixed_z = self.x_to_z(batch_data)
                print('batch_fixed_z')
                for batch_idx, node_idx in enumerate(current_nodes):
                    start = batch_idx * n_samples
                    end = (batch_idx + 1) * n_samples
                    parents = nodes_parents_map[node_idx]
                    for parent_idx in parents:
                        batch_z_orig[start:end, parent_idx] = batch_fixed_z[start:end, parent_idx]
                
                batch_sampled_x = self.z_to_x(batch_z_orig)
                print('batch_sampled_x')
                for batch_idx, node_idx in enumerate(current_nodes):
                    start = batch_idx * n_samples
                    end = (batch_idx + 1) * n_samples
                    
                    target_values = batch_sampled_x[start:end, node_idx].cpu().numpy()
                    
                    # 딕셔너리 초기화 (첫 quantile일 때)
                    if f'X{node_idx}|pa(X{node_idx})' not in sampled_from_cond:
                        sampled_from_cond[f'X{node_idx}|pa(X{node_idx})'] = {
                            'quantile_results': {},
                            'node_idx': node_idx,
                            'quantiles': quantiles
                        }
                    
                    sampled_from_cond[f'X{node_idx}|pa(X{node_idx})']['quantile_results'][f'q{q_idx}'] = {
                        'sampled_values': target_values
                    }

        print("=== Decompose_probability completed ===\n")
        return sampled_from_cond
    
    def _compute_conditional_prob_js(self, env1_samples, env2_samples, js_threshold,
                                      return_plot_data=False):
        """
        Compute KDE-based JS divergence and shift detection for each conditional
        distribution. Pure computation, no plotting.

        Args:
            env1_samples: Decomposed samples from environment 1
            env2_samples: Decomposed samples from environment 2
            js_threshold: Threshold for JS divergence shift detection
            return_plot_data: If True, also return per-quantile KDE arrays so the
                visualizer can render curves without recomputing.

        Returns:
            results: dict keyed by 'X{i}|pa(X{i})' with quantile_js_values,
                avg_js_divergence, shift_detected, node_idx.
            plot_data: dict with the same keys carrying KDE arrays for plotting,
                or None when return_plot_data=False.
        """
        results = {}
        plot_data = {} if return_plot_data else None

        for key in env1_samples.keys():
            node_idx = env1_samples[key]['node_idx']
            quantiles = env1_samples[key]['quantiles']
            n_quantiles = len(quantiles)

            quantile_js_values = []
            quantile_plots = [] if return_plot_data else None

            for q_idx in range(n_quantiles):
                q_key = f'q{q_idx}'
                samples1 = env1_samples[key]['quantile_results'][q_key]['sampled_values']
                samples2 = env2_samples[key]['quantile_results'][q_key]['sampled_values']

                samples1_finite = samples1[np.isfinite(samples1)]
                samples2_finite = samples2[np.isfinite(samples2)]

                if len(samples1_finite) == 0:
                    print(f"    ERROR: No finite values in env1 samples for node {node_idx} q{q_idx}")
                    continue
                if len(samples2_finite) == 0:
                    print(f"    ERROR: No finite values in env2 samples for node {node_idx} q{q_idx}")
                    continue

                var1 = np.var(samples1_finite)
                var2 = np.var(samples2_finite)

                # Degenerate-variance branch: use mean equality as a proxy for JS
                if var1 < 1e-10 or var2 < 1e-10:
                    if np.abs(np.mean(samples1_finite) - np.mean(samples2_finite)) < 1e-10:
                        js_div = 0
                    else:
                        js_div = float('inf')
                    quantile_js_values.append(js_div)
                    if return_plot_data:
                        quantile_plots.append({
                            'q_idx': q_idx,
                            'q_value': quantiles[q_idx],
                            'js_div': js_div,
                            'kde': None,
                        })
                    continue

                kde1 = gaussian_kde(samples1_finite, bw_method='scott')
                kde2 = gaussian_kde(samples2_finite, bw_method='scott')

                min_val = min(np.min(samples1_finite), np.min(samples2_finite))
                max_val = max(np.max(samples1_finite), np.max(samples2_finite))
                eval_range = np.linspace(min_val, max_val, 1000)

                pdf1 = kde1(eval_range)
                pdf2 = kde2(eval_range)
                pdf1_norm = pdf1 / np.sum(pdf1)
                pdf2_norm = pdf2 / np.sum(pdf2)

                js_div = jensenshannon(pdf1_norm, pdf2_norm)
                quantile_js_values.append(js_div)

                if return_plot_data:
                    quantile_plots.append({
                        'q_idx': q_idx,
                        'q_value': quantiles[q_idx],
                        'js_div': js_div,
                        'kde': {
                            'eval_range': eval_range,
                            'pdf1_norm': pdf1_norm,
                            'pdf2_norm': pdf2_norm,
                            'min_val': min_val,
                            'max_val': max_val,
                        },
                    })

            if len(quantile_js_values) == 0:
                print(f"    No valid quantiles processed for node {node_idx}")
                continue

            finite_js_values = [js for js in quantile_js_values if np.isfinite(js)]
            avg_js = np.mean(finite_js_values) if len(finite_js_values) > 0 else float('inf')

            results[key] = {
                'quantile_js_values': quantile_js_values,
                'avg_js_divergence': avg_js,
                'shift_detected': avg_js > js_threshold,
                'node_idx': node_idx,
            }

            if return_plot_data:
                plot_data[key] = {
                    'key': key,
                    'node_idx': node_idx,
                    'quantiles': quantiles,
                    'avg_js_divergence': avg_js,
                    'shift_detected': avg_js > js_threshold,
                    'js_threshold': js_threshold,
                    'quantile_plots': quantile_plots,
                }

        return results, plot_data

    def compare_conditional_probs(self, data1, data2, js_threshold=0.1, visualize=True, save_dir=None):

        quantiles = [0.25, 0.5, 0.75]
        env1_quantile_values = torch.quantile(data1, torch.tensor(quantiles).to(self.device), dim=0)
        print("\n=== Processing Environment 1 ===")
        env1_samples = self.decompose_probability(data1, quantile_values=env1_quantile_values)
        print("\n=== Processing Environment 2 ===")
        env2_samples = self.decompose_probability(data2, quantile_values=env1_quantile_values)

        use_viz = visualize and self.visualizer is not None
        results, plot_data = self._compute_conditional_prob_js(
            env1_samples, env2_samples, js_threshold, return_plot_data=use_viz
        )

        if use_viz:
            self.visualizer.plot_conditional_prob_comparison(plot_data, save_dir)

        avg_js_dict = {
            results[key]['node_idx']: results[key]['avg_js_divergence']
            for key in results.keys()
            if 'avg_js_divergence' in results[key]
        }

        shifted_nodes_threshold = [
            results[key]['node_idx']
            for key in results.keys()
            if 'shift_detected' in results[key] and results[key]['shift_detected']
        ]
        shifted_nodes_elbow = find_elbow_shift_nodes(avg_js_dict)

        results['shifted_nodes_threshold'] = shifted_nodes_threshold
        results['shifted_nodes_elbow'] = shifted_nodes_elbow
        results['shifted_nodes'] = shifted_nodes_threshold
        return results

    def dcor_score(self, x, y, random_state=42):
        np.random.seed(random_state)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1: x = x.reshape(-1, 1)
        return dcor.distance_correlation(x, y)
    
    def dcor_independence_test(self, x, y, random_state=42):
        np.random.seed(random_state)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1: x = x.reshape(-1, 1)        
        indep_results = dcor.independence.distance_covariance_test(x, y, num_resamples=500, random_state=random_state)
        p_value = indep_results.pvalue
        is_independent = p_value > 0.05

        return {
            'test_statistic': indep_results.statistic,
            'p_value': p_value,
            'is_independent': is_independent
        }

    def test_node_noise_independence(self, data1, data2, node_idx, save_dir=None, method='dcor'):
        """
        Test independence between a node's noise and its parent variables, comparing two environments
        Uses Distance Correlation test only
        
        Args:
            data1: Environment 1 data (numpy array)
            data2: Environment 2 data (numpy array, current environment)
            node_idx: Target node index
            method: 'dcor' (distance correlation only) or 'dcov_test' (distance covariance test with p-value)
            save_dir: Directory to save plots (optional)
        """
        parents = np.where(self.dag[node_idx,:] == 1)[0]
        node_noise_env1 = self.x_to_z(data1)[:, node_idx].cpu().numpy()  # Environment 1 noise
        node_noise_env2 = self.x_to_z(data2)[:, node_idx].cpu().numpy()  # Environment 2 noise (current)
        data1 = data1.cpu().numpy()
        data2 = data2.cpu().numpy()

        if self.visualizer is not None:
            self.visualizer.plot_noise_distribution_comparison(
                node_noise_env1, node_noise_env2, node_idx, save_dir
            )
        
        # === Distance Correlation Independence Tests ===
        parent_data_env1 = data1[:, parents]  # Shape: (n_samples, n_parents)
        parent_data_env2 = data2[:, parents]  # Shape: (n_samples, n_parents)

        if method == 'dcor':
            env1_dcor_score = self.dcor_score(parent_data_env1, node_noise_env1)
            env2_dcor_score = self.dcor_score(parent_data_env2, node_noise_env2)
            dcor_diff = env2_dcor_score - env1_dcor_score
            env1_test_results = {
                'dcor_score': env1_dcor_score
            }
            env2_test_results = {
                'dcor_score': env2_dcor_score
            }
            
        elif method == 'dcov_test':
            # Full distance covariance test with p-value (slower)
            env1_test_result = self.dcor_independence_test(parent_data_env1, node_noise_env1)            
            env2_test_result = self.dcor_independence_test(parent_data_env2, node_noise_env2)
            env1_test_result['dcor_score'] = None
            env2_test_result['dcor_score'] = None

        else:
            raise ValueError(f"Unknown method: {method}. Choose 'dcor' or 'dcov_test'")

        if self.visualizer is not None:
            self.visualizer.plot_parent_noise_scatter(
                parent_data_env1, parent_data_env2, node_noise_env1,
                node_noise_env2, parents, node_idx, save_dir
            )

        result = {
            'node_idx': node_idx,
            'parent_count': len(parents),
            'parents': list(parents),
            'env1_dcor_score': env1_dcor_score,
            'env2_dcor_score': env2_dcor_score,
        }

        if method == 'dcov_test':
            result.update({
                'env1_dcor_p': env1_test_result['p_value'],
                'env1_independent': env1_test_result['is_independent'],
                'env2_dcor_p': env2_test_result['p_value'],
                'env2_independent': env2_test_result['is_independent']
            })
    
        return result

    def conditional_standardization_gam(self, noise, parent_data, n_splines=20, lam=0.1, random_state=42):
        """
        Standardize noise using GAM (Generalized Additive Models) for conditional mean and variance estimation.
        Non-parametric approach with smooth spline fitting including interaction terms.
        
        Args:
            noise: Node noise values (1D array)
            parent_data: Parent variable values (2D array: n_samples x n_parents)
            n_splines: Number of splines for GAM (default: 20)
            lam: Smoothing parameter for GAM (default: 0.1, higher = smoother)
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Standardized residuals and GAM fitting results
        """
        
        print(f"    Fitting GAM with n_splines={n_splines}, lambda={lam}...")
        
        if parent_data.ndim == 1:
            parent_data = parent_data.reshape(-1, 1)
        
        n_samples, n_parents = parent_data.shape
        np.random.seed(random_state)
        scaler = StandardScaler()
        parent_data_scaled = scaler.fit_transform(parent_data)
        
        # 1. Fit conditional mean using GAM with interaction terms
        if n_parents == 1:
            gam_mean = GAM(s(0, n_splines=n_splines, lam=lam))
        else:
            # Add all additive terms + all pairwise interaction terms
            terms = [s(i, n_splines=n_splines, lam=lam) for i in range(n_parents)]
            for i in range(n_parents):
                for j in range(i + 1, n_parents):
                    terms.append(te(i, j, lam=lam))
            gam_mean = GAM(reduce(operator.add, terms))
            print(f"      Model includes {n_parents * (n_parents - 1) // 2} pairwise interactions")
        
        gam_mean.fit(parent_data_scaled, noise)
        cond_mean = gam_mean.predict(parent_data_scaled)
        residuals = noise - cond_mean

        squared_residuals = residuals ** 2

        if n_parents == 1:
            gam_var = GAM(s(0, n_splines=n_splines, lam=lam))
        else:
            terms = [s(i, n_splines=n_splines, lam=lam) for i in range(n_parents)]
            for i in range(n_parents):
                for j in range(i + 1, n_parents):
                    terms.append(te(i, j, lam=lam))
            gam_var = GAM(reduce(operator.add, terms))
        
        gam_var.fit(parent_data_scaled, squared_residuals)

        # Predict variance
        cond_var = gam_var.predict(parent_data_scaled)

        # Ensure positive variance
        global_var = np.var(residuals)
        var_floor = max(0.01, global_var * 0.01)
        cond_var = np.maximum(cond_var, var_floor)
        cond_std = np.sqrt(cond_var)

        # Standardize residuals
        standardized_residuals = residuals / cond_std
        
        # Calculate R² for mean fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((noise - np.mean(noise)) ** 2)
        mean_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Variance R²
        abs_residuals = np.abs(residuals)
        var_pred = cond_std
        var_ss_res = np.sum((abs_residuals - var_pred) ** 2)
        var_ss_tot = np.sum((abs_residuals - np.mean(abs_residuals)) ** 2)
        var_r2 = 1 - (var_ss_res / var_ss_tot) if var_ss_tot > 0 else 0
        
        print(f"      Mean fit R²: {mean_r2:.4f}, Variance fit R²: {var_r2:.4f}")
        
        return {
            'standardized_residuals': standardized_residuals,
            'squared_residuals': squared_residuals,
            'residuals': residuals,
            'cond_mean': cond_mean,
            'cond_var': cond_var,
            'cond_std': cond_std,
            'gam_mean': gam_mean,
            'gam_var': gam_var,
            'scaler': scaler,
            'mean_r2': mean_r2,
            'var_r2': var_r2,
            'n_parents': n_parents,
            'n_splines': n_splines,
            'lam': lam,
            'parent_data': parent_data,
            'parent_data_scaled': parent_data_scaled,
            'noise': noise
        }
    
    def chisquare_goodness_of_fit(self, standardized_residuals, alpha=0.05):        
        squared_residuals = standardized_residuals ** 2
        ks_stat, ks_pvalue = kstest(squared_residuals, lambda x: chi2.cdf(x, df=1))
        is_similar_to_chisq = ks_pvalue > alpha
        
        print(f"    KS statistic: {ks_stat:.4f}")
        print(f"    KS p-value: {ks_pvalue:.4f}")
        print(f"    Similar to χ²(1): {is_similar_to_chisq} (alpha={alpha})")

        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'is_similar_to_chisq': is_similar_to_chisq,
            'squared_residuals': squared_residuals,
            'alpha': alpha
        }
    
    def test_simultaneous_shift(self, data2, node_idx, save_dir=None, n_splines=20, lam=0.1):
        """
        Test for simultaneous shift (function + noise) vs function-only shift.
        Uses GAM (Generalized Additive Models) for robust non-parametric estimation.
        
        Process:
        1. Extract noise from environment 2
        2. Conditional standardization using GAM
        3. Chi-square goodness-of-fit test
        
        Args:
            data2: Environment 2 data (numpy array)
            node_idx: Target node index
            save_dir: Directory to save plots
            n_splines: Number of splines for GAM (default: 10)
            lam: Smoothing parameter for GAM (default: 0.6)
            
        Returns:
            dict: Test results and shift classification
        """
        print(f"\n=== Testing Simultaneous Shift for Node {node_idx} (GAM) ===")
        parents = np.where(self.dag[node_idx,:] == 1)[0]
        print(f"  Parents: {list(parents)}")
        
        # Extract noise from environment 2
        node_noise_env2 = self.x_to_z(data2)[:, node_idx].cpu().numpy()
        data2 = data2.cpu().numpy()
        parent_data_env2 = data2[:, parents]
        # Conditional standardization using GAM
        standardization_result = self.conditional_standardization_gam(
            node_noise_env2, parent_data_env2, 
            n_splines=n_splines,
            lam=lam
        )
        
        if self.visualizer is not None:
            self.visualizer.plot_gam_fitting(standardization_result, node_idx, parents, save_dir)
            self.visualizer.plot_gam_diagnostics(standardization_result, node_idx, parents, save_dir)

        # Chi-square goodness-of-fit test
        chisq_test_result = self.chisquare_goodness_of_fit(
            standardization_result['standardized_residuals']
        )
        
        # Determine shift type
        if chisq_test_result['is_similar_to_chisq']:
            shift_type = 'function_only'
            explanation = "Squared standardized residuals follow χ²(1) → Function shift only"
        else:
            shift_type = 'function_and_noise'
            explanation = "Squared standardized residuals deviate from χ²(1) → Function + Noise shift"
        
        print(f"  Shift Type: {shift_type}")
        print(f"  Explanation: {explanation}")
        print(f"  Mean fit R²: {standardization_result['mean_r2']:.4f}")
        print(f"  Variance fit R²: {standardization_result['var_r2']:.4f}")
        
        if self.visualizer is not None:
            self.visualizer.plot_simultaneous_shift_test(
                standardization_result, chisq_test_result, node_idx, shift_type, save_dir
            )
        
        return {
            'node_idx': node_idx,
            'parents': list(parents),
            'shift_type': shift_type,
            'explanation': explanation,
            'ks_statistic': chisq_test_result['ks_statistic'],
            'ks_pvalue': chisq_test_result['ks_pvalue'],
            'is_similar_to_chisq': chisq_test_result['is_similar_to_chisq'],
            'mean_r2': standardization_result['mean_r2'],
            'var_r2': standardization_result['var_r2'],
            'n_splines': n_splines,
            'lam': lam
        }

    def analyze(self, data1, data2, save_dir, simultaneous_shift=False, shifted_nodes=None, independence_method='dcor'):
        detection_start_time = time.time()
        self.save_dir = save_dir
        self.independence_method = independence_method
        data1 = data1.to(self.device)
        data2 = data2.to(self.device)        

        # Select only first 10 samples from each dataset
        print(f"Environment 1 data shape: {data1.shape}")
        print(f"Environment 2 data shape: {data2.shape}")
        
        comparison_results = self.compare_conditional_probs(
            data1, data2, js_threshold=0.1, visualize=True, save_dir=self.save_dir
        )
        print(f"Analyzing nodes: {shifted_nodes}")
        
        if shifted_nodes is None:
            shifted_nodes = comparison_results['shifted_nodes']
            print('Comparsion results are used for shifted nodes')
        detection_end_time = time.time()
        dissection_start_time = time.time()
        # Run independence tests for detected/specified nodes
        independence_results = {}
        simultaneous_shift_results = {}
        dcor_score = {}
        dcor_threshold = 0.05
        estimated_shift_types = {}

        for node_idx in shifted_nodes:
            independence_result = self.test_node_noise_independence(
                data1, data2, node_idx, save_dir=self.save_dir, method=independence_method
            )
            independence_results[node_idx] = independence_result
            dcor_score[node_idx] = independence_result['env2_dcor_score'] - independence_result['env1_dcor_score']

            if independence_method == 'dcov_test':
                if independence_results[node_idx]['env2_independent']: shift_types = 'noise'
                elif (independence_results[node_idx]['env2_independent'] == False): shift_types = 'function'
                else: 
                    raise ValueError(f"Unexpected independence test result for node {node_idx}: {independence_results[node_idx]['env2_independent']}")
                estimated_shift_types[str(node_idx)] = shift_types

            if node_idx in dcor_score and dcor_score[node_idx] > dcor_threshold: shift_types = 'function'
            else: shift_types = 'noise'
            estimated_shift_types[str(node_idx)] = shift_types

            # Simultaneous shift test if enabled
            if simultaneous_shift:
                simultaneous_result = self.test_simultaneous_shift(
                    data2, node_idx, save_dir=self.save_dir
                )
                simultaneous_shift_results[node_idx] = simultaneous_result
                
                # Integrate results for final classification
                final_classification = self._classify_shift_type(
                    independence_result, simultaneous_result
                )
                
                print(f"\n=== Final Classification for Node {node_idx} ===")
                print(f"  Shift Type: {final_classification['shift_type']}")
                print(f"  Explanation: {final_classification['explanation']}")
        
        independence_results['estimated_shift_types'] = estimated_shift_types
        dissection_end_time = time.time()
        return {
            'comparison_results': comparison_results,
            'independence_results': independence_results,
            'simultaneous_shift_results': simultaneous_shift_results if simultaneous_shift else None,
            'data_shapes': {
                'data1': data1.shape,
                'data2': data2.shape
            },
            'num_samples': data1.shape[0],            
            'save_dir': self.save_dir,
            'analyzed_nodes': shifted_nodes,
            'detection_time': detection_end_time - detection_start_time,
            'dissection_time': dissection_end_time - dissection_start_time
        }
    
    def _classify_shift_type(self, independence_result, simultaneous_result):
        """
        Classify shift type based on independence test and chi-square test.
        
        Logic:
        - Independence test: Distinguishes function shift vs noise shift
        - Chi-square test: Distinguishes function-only vs function+noise shift
        """
        if self.independence_method == 'dcor':
            if simultaneous_result['is_similar_to_chisq']:
                return {
                    'shift_type': 'function_only',
                    'explanation': 'χ²(1) match → Function shift only'
                }
            else:
                return {
                    'shift_type': 'function_and_noise',
                    'explanation': 'χ²(1) mismatch → Function + Noise shift'
                }
        

        env2_independent = independence_result['env2_independent']
        if env2_independent:
            # Noise is independent → Noise shift (noise distribution changed)
            return {
                'shift_type': 'noise_shift_dominant',
                'explanation': 'Independent noise → Noise shift (noise distribution changed)'
            }
        else:
            # Noise is dependent → Function shift (possibly with noise change)
            if simultaneous_result['is_similar_to_chisq']:
                return {
                    'shift_type': 'function_only',
                    'explanation': 'Dependent noise + χ²(1) match → Function shift only'
                }
            else:
                return {
                    'shift_type': 'function_and_noise',
                    'explanation': 'Dependent noise + χ²(1) mismatch → Function + Noise shift'
                }
    