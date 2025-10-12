import os
import math
import torch
import numpy as np
from scipy import stats
from scipy.stats import chi2, kstest
import dcor
from functools import reduce
import operator
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import GAM, s, te
from visualize import FANSVisualizer

class FANSAnalyzer:
    """
    FANS (Flow-based Analysis of Noise Shift) implementation
    Extracted from CausalNFightning for modular use
    """

    def __init__(self, model, preparator, device, input_scaler=None, external_dag_path=None):
        """
        Initialize FANS analyzer
        
        Args:
            model: Trained causal normalizing flow model
            preparator: Data preparator with adjacency matrix and data generation methods
            device: PyTorch device (CPU/GPU)
            input_scaler: Input scaler
            external_dag_path: Path to external DAG
        """
        self.model = model
        self.preparator = preparator
        self.device = device
        self.external_dag_path = external_dag_path
        self.input_scaler = input_scaler
        self.visualizer = FANSVisualizer()

    def get_x_norm(self, batch):
        """Get normalized x from batch (helper method)"""
        x_norm = self.input_scaler.transform(batch[0].to(self.device), inplace=False)
        return x_norm
    
    def x_to_z(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.astype(np.float32))
        x = x.to(self.device)
        
        # Convert to batch format if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_norm = self.get_x_norm((x,))
        
        # Forward transform
        with torch.no_grad():
            z = self.model.flow().transform(x_norm)
        return z 
    
    def z_to_x(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z.astype(np.float32))
        z = z.to(self.device)
        
        # Inverse transform
        with torch.no_grad():
            x_norm = self.model.flow().transform.inv(z)
            
        # Use input_scaler's inverse_transform
        x = self.input_scaler.inverse_transform(x_norm, inplace=False)
        return x 

    def decompose_probability(self, data, quantile_values=None):
        print("=== Starting GPU-optimized decompose_probability ===")
        dag = self.preparator.adjacency(False).cpu().numpy()        
        z_orig = self.x_to_z(data)
        
        n_vars = data.shape[1]
        quantiles = [0.25, 0.5, 0.75]
        
        if quantile_values is not None:
            if not isinstance(quantile_values, torch.Tensor):
                quantile_values = torch.tensor(quantile_values.astype(np.float32)).to(self.device)
            else:
                quantile_values = quantile_values.to(self.device)
        
        print("Pre-computing all quantile latent representations...")
        quantile_z_cache = {}
        
        for q_idx, q_value in enumerate(quantiles):
            print(f"  Pre-computing quantile {q_idx} (p={q_value})...")
            
            fixed_data = data.clone()
            for var_idx in range(n_vars):
                fixed_data[:, var_idx] = quantile_values[q_idx, var_idx]
            
            fixed_z = self.x_to_z(fixed_data)
            quantile_z_cache[q_idx] = fixed_z
            
            print(f"    Cached quantile {q_idx} latent representation")
        
        print("Pre-computation completed. Processing nodes...")
        
        sampled_from_cond = {}
        
        for node_idx in range(n_vars):
            parents = np.where(dag[node_idx,:] == 1)[0]
            
            if len(parents) > 0:
                print(f"\n--- Processing Node {node_idx} (parents: {list(parents)}) ---")
                quantile_results = {}
                
                for q_idx, q_value in enumerate(quantiles):
                    print(f"  Processing quantile {q_idx} for node {node_idx}...")
                    
                    # Start with original data
                    combined_z = z_orig.clone()
                    
                    # Apply quantile condition only to parent nodes
                    for parent_idx in parents:
                        combined_z[:, parent_idx] = quantile_z_cache[q_idx][:, parent_idx]
                    
                    # Transform back to data space from GPU
                    sampled_x = self.z_to_x(combined_z)
                    
                    # Move to CPU only at this point for KDE computation
                    target_values = sampled_x[:, node_idx].cpu().numpy()
                    
                    quantile_results[f'q{q_idx}'] = {
                        'sampled_values': target_values,
                        'quantile_value': quantile_values[q_idx].cpu().numpy()
                    }

                sampled_from_cond[f'X{node_idx}|pa(X{node_idx})'] = {
                    'quantile_results': quantile_results,
                    'node_idx': node_idx,
                    'quantiles': quantiles
                }

        print("=== GPU-optimized decompose_probability completed ===\n")
        return sampled_from_cond
    
    def compare_conditional_probs(self, data1, data2, js_threshold=0.1, visualize=True, save_dir=None):
                
        quantiles = [0.25, 0.5, 0.75]
        env1_quantile_values = torch.quantile(data1, torch.tensor(quantiles).to(self.device), dim=0)
            
        print("Environment 1 quantile values:")
        print(env1_quantile_values.cpu().numpy())
            
        # Generate samples for each environment using env1's quantiles for both
        print("\n=== Processing Environment 1 ===")
        env1_samples = self.decompose_probability(data1, quantile_values=env1_quantile_values)
        print("\n=== Processing Environment 2 ===")
        env2_samples = self.decompose_probability(data2, quantile_values=env1_quantile_values)
            
        results = {}
        
        if visualize:
            # Delegate visualization to FANSVisualizer
            results = self.visualizer.plot_conditional_prob_comparison(
                env1_samples, env2_samples, js_threshold, save_dir
            )
        
        # Identify nodes with detected shifts
        shifted_nodes = [results[key]['node_idx'] for key in results.keys() 
                        if 'shift_detected' in results[key] and results[key]['shift_detected']]    
        
        results['shifted_nodes'] = shifted_nodes
        return results

    def dcor_independence_test(self, x, y, random_state=42):
        """
        Distance Correlation based independence test using dcor.independence
        
        Args:
            x: Parent variable values (1D or 2D array)
            y: Node noise values (1D array)
            random_state: Random seed
            
        Returns:
            dict: dcor score, p-value, and independence decision
        """
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Ensure consistent data types (float64) for dcor
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Ensure proper format - dcor can handle both 1D and 2D arrays
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Calculate distance correlation
        dcor_score = dcor.distance_correlation(x, y)
        
        # Perform independence test using dcor.independence
        indep_results = dcor.independence.distance_covariance_test(x, y, num_resamples=500)
        p_value = indep_results.pvalue

        # Independence decision (α = 0.05)
        is_independent = p_value > 0.05
        
        return {
            'dcor_score': dcor_score,
            'test_statistic': indep_results.statistic,
            'p_value': p_value,
            'is_independent': is_independent
        }

    def test_independence_dcor_only(self, parent_values, noise_values):
        """
        Distance Correlation only independence test
        
        Args:
            parent_values: All parent variable values (2D array: n_samples x n_parents)
            noise_values: Node noise values (1D array)
            
        Returns:
            dict: dcor independence test results
        """
        # Calculate dcor independence test
        dcor_result = self.dcor_independence_test(parent_values, noise_values)
        
        # Independence decision based on dcor
        is_independent = dcor_result['is_independent']
        
        return {
            'dcor_score': dcor_result['dcor_score'],
            'test_statistic': dcor_result['test_statistic'],
            'dcor_p': dcor_result['p_value'],
            'is_independent': is_independent,
            'dependency_type': "Independent" if is_independent else "Dependent"
        }

    def test_node_noise_independence(self, data1, data2, node_idx, save_dir=None):
        """
        Test independence between a node's noise and its parent variables, comparing two environments
        Uses Distance Correlation test only
        
        Args:
            data1: Environment 1 data (numpy array)
            data2: Environment 2 data (numpy array, current environment)
            node_idx: Target node index
            save_dir: Directory to save plots (optional)
        """
        dag = self.preparator.adjacency(False).cpu().numpy()
        parents = np.where(dag[node_idx,:] == 1)[0]
        
        if len(parents) == 0:
            return {
                'node_idx': node_idx,
                'parent_count': 0,
                'env1_dcor_score': 0.0,
                'env1_dcor_p': 1.0,
                'env1_independent': True,
                'env2_dcor_score': 0.0,
                'env2_dcor_p': 1.0,
                'env2_independent': True,
                'dependency_type_env1': "Independent",
                'dependency_type_env2': "Independent"
            }
        
        node_noise_env1 = self.x_to_z(data1)[:, node_idx].cpu().numpy()  # Environment 1 noise
        node_noise_env2 = self.x_to_z(data2)[:, node_idx].cpu().numpy()  # Environment 2 noise (current)

        # Visualize noise distributions
        self.visualizer.plot_noise_distribution_comparison(
            node_noise_env1, node_noise_env2, node_idx, save_dir
        )
        
        # === Distance Correlation Independence Tests ===
        parent_data_env1 = data1[:, parents]  # Shape: (n_samples, n_parents)
        parent_data_env2 = data2[:, parents]  # Shape: (n_samples, n_parents)

        env1_test_result = self.test_independence_dcor_only(parent_data_env1, node_noise_env1)
        env2_test_result = self.test_independence_dcor_only(parent_data_env2, node_noise_env2)

        # Visualize parent-noise scatter plots
        self.visualizer.plot_parent_noise_scatter(
            parent_data_env1, parent_data_env2, node_noise_env1, 
            node_noise_env2, parents, node_idx, save_dir
        )

        # Visualize DCor independence test
        self.visualizer.plot_dcor_independence_test(
            parent_data_env1, parent_data_env2, node_noise_env1, 
            node_noise_env2, parents, node_idx, env1_test_result, 
            env2_test_result, save_dir
        )

        result = {
            'node_idx': node_idx,
            'parent_count': len(parents),
            'parents': list(parents),
            'env1_dcor_score': env1_test_result['dcor_score'],
            'env1_dcor_p': env1_test_result['dcor_p'],
            'env1_independent': env1_test_result['is_independent'],
            'env2_dcor_score': env2_test_result['dcor_score'],
            'env2_dcor_p': env2_test_result['dcor_p'],
            'env2_independent': env2_test_result['is_independent'],
            'dependency_type_env1': env1_test_result['dependency_type'],
            'dependency_type_env2': env2_test_result['dependency_type']
        }
        return result

    def conditional_standardization_gam(self, noise, parent_data, n_splines=20, lam=0.1, random_state=42):
        """
        Standardize noise using GAM (Generalized Additive Models) for conditional mean and variance estimation.
        Non-parametric approach with smooth spline fitting including interaction terms.
        Outliers (top/bottom 2.5% of residuals) are removed.
        
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
        
        # Ensure proper shapes
        if parent_data.ndim == 1:
            parent_data = parent_data.reshape(-1, 1)
        
        n_samples, n_parents = parent_data.shape
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Standardize parent data for GAM fitting (improves numerical stability)
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
        
        # Predict conditional mean
        cond_mean = gam_mean.predict(parent_data_scaled)
        
        # Calculate residuals
        residuals = noise - cond_mean
        
        # Remove outliers: top 2.5% and bottom 2.5%
        lower_bound = np.percentile(residuals, 2.5)
        upper_bound = np.percentile(residuals, 97.5)
        
        inlier_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
        n_outliers = np.sum(~inlier_mask)
        print(f"      Outliers removed: {n_outliers}/{len(residuals)} ({n_outliers/len(residuals)*100:.1f}%)")
        
        # Filter all data based on inlier mask
        residuals = residuals[inlier_mask]
        noise = noise[inlier_mask]
        parent_data = parent_data[inlier_mask]
        parent_data_scaled = parent_data_scaled[inlier_mask]
        cond_mean = cond_mean[inlier_mask]
        
        # 2. Fit conditional variance using GAM on SQUARED residuals with interactions
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
        """
        Test if squared standardized residuals follow chi-square(1) distribution.
        
        Args:
            standardized_residuals: Standardized residuals (1D array)
            alpha: Significance level (default: 0.05)
            
        Returns:
            dict: Test results including KS test and visual comparison
        """
        print("    Performing chi-square(1) goodness-of-fit test...")
        
        # Square the standardized residuals
        squared_residuals = standardized_residuals ** 2
        
        # Kolmogorov-Smirnov test against chi-square(1)
        ks_stat, ks_pvalue = kstest(squared_residuals, lambda x: chi2.cdf(x, df=1))
        
        # Decision: if p-value > alpha, distributions are similar (function shift only)
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
        
        dag = self.preparator.adjacency(False).cpu().numpy()
        parents = np.where(dag[node_idx,:] == 1)[0]
        
        if len(parents) == 0:
            print(f"  Node {node_idx} has no parents. Skipping simultaneous shift test.")
            return {
                'node_idx': node_idx,
                'has_parents': False,
                'shift_type': 'noise_only'
            }
        
        print(f"  Parents: {list(parents)}")
        
        # Extract noise from environment 2
        node_noise_env2 = self.x_to_z(data2)[:, node_idx].cpu().numpy()
        parent_data_env2 = data2[:, parents]
        
        # Conditional standardization using GAM
        standardization_result = self.conditional_standardization_gam(
            node_noise_env2, parent_data_env2, 
            n_splines=n_splines,
            lam=lam
        )
        
        # Plot GAM fitting (delegated to visualizer)
        self.visualizer.plot_gam_fitting(standardization_result, node_idx, parents, save_dir)
        
        # Plot GAM diagnostics (delegated to visualizer)
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
        
        # Visualization (delegated to visualizer)
        self.visualizer.plot_simultaneous_shift_test(
            standardization_result, chisq_test_result, node_idx, shift_type, save_dir
        )
        
        return {
            'node_idx': node_idx,
            'has_parents': True,
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

    def analyze(self, data1, data2, save_dir=None, simultaneous_shift=False, shifted_nodes=None):
        self.save_dir = save_dir

        # Set save directory
        if save_dir is None: save_dir = "fans_analysis"
        print("Generating data from Environment 1")
        if isinstance(data1, torch.Tensor): 
            data1 = data1.to(self.device)
        else:
            data1 = torch.tensor(data1.astype(np.float32)).to(self.device)
        
        if isinstance(data2, torch.Tensor): 
            data2 = data2.to(self.device)
        else:
            data2 = torch.tensor(data2.astype(np.float32)).to(self.device)
        
        print("Successfully loaded both environments on GPU")
        print(f"Environment 1 data shape: {data1.shape}")
        print(f"Environment 2 data shape: {data2.shape}")
        
        # Compare conditional distributions if shifted_nodes not provided
        
        comparison_results = self.compare_conditional_probs(
            data1, data2, js_threshold=0.1, visualize=True, save_dir=self.save_dir
        )
        
        print(f"Analyzing nodes: {shifted_nodes}")
        
        # Run independence tests for detected/specified nodes
        independence_results = {}
        simultaneous_shift_results = {}
        
        for node_idx in shifted_nodes:
            independence_result = self.test_node_noise_independence(
                data1.cpu().numpy(), data2.cpu().numpy(), node_idx, save_dir=self.save_dir
            )
            independence_results[node_idx] = independence_result
            
            print(f"\nNode {node_idx} Distance Correlation Independence Test:")
            print(f"  Parents: {independence_result['parents']}")
            print(f"  Environment 1:")
            print(f"    DCor Score: {independence_result['env1_dcor_score']:.4f}")
            print(f"    DCor p-value: {independence_result['env1_dcor_p']:.4f}")
            print(f"    Independent: {independence_result['env1_independent']}")
            print(f"  Environment 2:")
            print(f"    DCor Score: {independence_result['env2_dcor_score']:.4f}")
            print(f"    DCor p-value: {independence_result['env2_dcor_p']:.4f}")
            print(f"    Independent: {independence_result['env2_independent']}")
            
            # Simultaneous shift test if enabled
            if simultaneous_shift:
                simultaneous_result = self.test_simultaneous_shift(
                    data2.cpu().numpy(), node_idx, save_dir=self.save_dir
                )
                simultaneous_shift_results[node_idx] = simultaneous_result
                
                # Integrate results for final classification
                final_classification = self._classify_shift_type(
                    independence_result, simultaneous_result
                )
                
                print(f"\n=== Final Classification for Node {node_idx} ===")
                print(f"  Shift Type: {final_classification['shift_type']}")
                print(f"  Explanation: {final_classification['explanation']}")
        
        # Generate final summary if simultaneous_shift is enabled
        if simultaneous_shift:
            self._generate_shift_summary(
                shifted_nodes, independence_results, simultaneous_shift_results, save_dir
            )

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
            'analyzed_nodes': shifted_nodes
        }
    
    def _classify_shift_type(self, independence_result, simultaneous_result):
        """
        Classify shift type based on independence test and chi-square test.
        
        Logic:
        - Independence test: Distinguishes function shift vs noise shift
        - Chi-square test: Distinguishes function-only vs function+noise shift
        """
        env2_independent = independence_result['env2_independent']
        
        if not simultaneous_result['has_parents']:
            return {
                'shift_type': 'noise_only',
                'explanation': 'Node has no parents → Only noise shift possible'
            }
        
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
    
    def _generate_shift_summary(self, shifted_nodes, independence_results, 
                                 simultaneous_shift_results, save_dir):
        """
        Generate summary table and visualization of shift classification.
        """
        print("\n" + "="*80)
        print("SHIFT CLASSIFICATION SUMMARY")
        print("="*80)
        
        summary_data = []
        
        for node_idx in shifted_nodes:
            indep_result = independence_results[node_idx]
            simul_result = simultaneous_shift_results.get(node_idx, {})
            
            if simul_result.get('has_parents', True):
                final_class = self._classify_shift_type(indep_result, simul_result)
                
                summary_data.append({
                    'Node': f'X{node_idx}',
                    'Parents': indep_result.get('parents', []),
                    'Env2 Independent': indep_result['env2_independent'],
                    'χ²(1) Similar': simul_result.get('is_similar_to_chisq', 'N/A'),
                    'KS p-value': f"{simul_result.get('ks_pvalue', 0):.4f}",
                    'Shift Type': final_class['shift_type']
                })
                
                print(f"\nNode X{node_idx}:")
                print(f"  Parents: {indep_result.get('parents', [])}")
                print(f"  Env2 Independent: {indep_result['env2_independent']}")
                print(f"  χ²(1) Similar: {simul_result.get('is_similar_to_chisq', 'N/A')}")
                print(f"  → Shift Type: {final_class['shift_type']}")
            else:
                summary_data.append({
                    'Node': f'X{node_idx}',
                    'Parents': [],
                    'Env2 Independent': 'N/A',
                    'χ²(1) Similar': 'N/A',
                    'KS p-value': 'N/A',
                    'Shift Type': 'noise_only'
                })
        
        # Create summary DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        summary_file = os.path.join(save_dir, 'shift_classification_summary.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"\nSaved shift classification summary: {summary_file}")
        
        # Display summary table
        print("\n" + df_summary.to_string(index=False))
        print("="*80)
 