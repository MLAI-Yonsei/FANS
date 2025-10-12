import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, chi2, norm
from scipy.spatial.distance import jensenshannon
import pandas as pd


class FANSVisualizer:
    """
    Visualization utilities for FANS (Flow-based Analysis of Noise Shift)
    Separated from FANSAnalyzer for modularity
    """
    
    @staticmethod
    def plot_conditional_prob_comparison(env1_samples, env2_samples, js_threshold, save_dir):
        """
        Visualize conditional probability comparison between two environments.
        
        Args:
            env1_samples: Decomposed samples from environment 1
            env2_samples: Decomposed samples from environment 2
            js_threshold: Threshold for JS divergence
            save_dir: Directory to save plots
            
        Returns:
            dict: Results including JS divergence values and shift detection
        """
        results = {}
        
        for key in env1_samples.keys():
            node_idx = env1_samples[key]['node_idx']
            quantiles = env1_samples[key]['quantiles']
            print(f"\n=== Visualizing Node {node_idx} ===")
            
            # Calculate subplot layout
            n_quantiles = len(quantiles)
            n_cols = min(2, n_quantiles)
            n_rows = math.ceil(n_quantiles / n_cols)
            
            plt.figure(figsize=(n_cols*6, n_rows*4))
            plt.suptitle(f'Distribution Comparison for {key}: Env1 vs Env2', 
                        fontsize=16, fontweight='bold')
            
            # Create separate subplot for each quantile
            quantile_js_values = []
            
            for q_idx in range(n_quantiles):
                q_key = f'q{q_idx}'
                
                print(f"  Processing quantile {q_idx}...")
                samples1 = env1_samples[key]['quantile_results'][q_key]['sampled_values']
                samples2 = env2_samples[key]['quantile_results'][q_key]['sampled_values']
                
                # Check for problematic values before KDE
                samples1_finite = samples1[np.isfinite(samples1)]
                samples2_finite = samples2[np.isfinite(samples2)]
                
                if len(samples1_finite) == 0:
                    print(f"    ERROR: No finite values in env1 samples for quantile {q_idx}")
                    continue
                if len(samples2_finite) == 0:
                    print(f"    ERROR: No finite values in env2 samples for quantile {q_idx}")
                    continue
                
                # Check variance
                var1 = np.var(samples1_finite)
                var2 = np.var(samples2_finite)
                print(f"    Variance: env1={var1:.6f}, env2={var2:.6f}")
                
                if var1 < 1e-10:
                    print(f"    WARNING: Very low variance in env1 samples")
                if var2 < 1e-10:
                    print(f"    WARNING: Very low variance in env2 samples")
                
                # Estimate distribution using KDE
                print(f"    Computing KDE for env1...")
                kde_samples1 = gaussian_kde(samples1_finite, bw_method='scott')
                print(f"    Computing KDE for env2...")
                kde_samples2 = gaussian_kde(samples2_finite, bw_method='scott')
                
                # Set evaluation range
                min_val = min(np.min(samples1_finite), np.min(samples2_finite))
                max_val = max(np.max(samples1_finite), np.max(samples2_finite))
                eval_range = np.linspace(min_val, max_val, 1000)
                
                # Calculate PDF using KDE
                kde_pdf_samples1 = kde_samples1(eval_range)
                kde_pdf_norm_samples1 = kde_pdf_samples1 / np.sum(kde_pdf_samples1)
                kde_pdf_samples2 = kde_samples2(eval_range)
                kde_pdf_norm_samples2 = kde_pdf_samples2 / np.sum(kde_pdf_samples2)
                
                # Calculate Jensen-Shannon divergence
                js_div = jensenshannon(kde_pdf_norm_samples1, kde_pdf_norm_samples2)
                quantile_js_values.append(js_div)
                
                print(f"    JS divergence: {js_div:.6f}")
                
                # Visualize KDE for each quantile
                plt.subplot(n_rows, n_cols, q_idx + 1)
                plt.plot(eval_range, kde_pdf_norm_samples1, 'b-', label='Env1')
                plt.plot(eval_range, kde_pdf_norm_samples2, 'r-', label='Env2')
                
                plt.title(f'Quantile {q_idx} (p={quantiles[q_idx]}): JS={js_div:.4f}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                
                # Highlight background if threshold exceeded
                if js_div > js_threshold:
                    plt.axvspan(min_val, max_val, alpha=0.1, color='red')
            
            if len(quantile_js_values) > 0:
                # Calculate average JS divergence
                avg_js = np.mean(quantile_js_values)
                
                # Store results
                results[key] = {
                    'quantile_js_values': quantile_js_values,
                    'avg_js_divergence': avg_js,
                    'shift_detected': avg_js > js_threshold,
                    'node_idx': node_idx
                }
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                
                # Get dataset name for filename
                filename = f"conditional_prob_comparison_node_{node_idx}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved visualization: {filepath}")
                plt.show()
                plt.close()
            else:
                print(f"    No valid quantiles processed for node {node_idx}")
                plt.close()
        
        return results
    
    @staticmethod
    def plot_noise_distribution_comparison(node_noise_env1, node_noise_env2, node_idx, save_dir):
        """
        Plot noise distribution comparison between two environments.
        """
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Noise distributions
        plt.subplot(1, 3, 1)
        plt.hist(node_noise_env1, bins=50, alpha=0.7, label='Environment 1', color='blue', density=True)
        plt.hist(node_noise_env2, bins=50, alpha=0.7, label='Environment 2', color='red', density=True)
        plt.xlabel(f'Noise values for Node X{node_idx}')
        plt.ylabel('Density')
        plt.title(f'Noise Distribution Comparison\nNode X{node_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Q-Q plot
        plt.subplot(1, 3, 2)
        stats.probplot(node_noise_env1, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot: Env1 Noise\nNode X{node_idx}')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Q-Q plot for env2
        plt.subplot(1, 3, 3)
        stats.probplot(node_noise_env2, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot: Env2 Noise\nNode X{node_idx}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"noise_distribution_comparison_node_{node_idx}.png"
        
        if save_dir is not None:
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = filename
    
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved noise distribution comparison: {filepath}")
        
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_parent_noise_scatter(parent_data_env1, parent_data_env2, node_noise_env1, 
                                   node_noise_env2, parents, node_idx, save_dir):
        """
        Plot individual parent-noise relationships for all parents.
        """
        n_parents = len(parents)
        n_cols = min(3, n_parents)
        n_rows = math.ceil(n_parents / n_cols)
        
        # Figure: Individual parent scatter plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_parents == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_parents > 1 else axes
        
        for i, parent_idx in enumerate(parents):
            ax = axes[i]
            ax.scatter(parent_data_env1[:, i], node_noise_env1, alpha=0.3, s=5, color='blue', label='Env1')
            ax.scatter(parent_data_env2[:, i], node_noise_env2, alpha=0.3, s=5, color='red', label='Env2')
            ax.set_xlabel(f'Parent X{parent_idx} values')
            ax.set_ylabel(f'Node X{node_idx} noise')
            ax.set_title(f'Parent X{parent_idx} vs Noise')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_parents, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Individual Parent-Noise Relationships: Node X{node_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"parent_noise_scatter_node_{node_idx}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved parent-noise scatter plots: {filepath}")
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_dcor_independence_test(parent_data_env1, parent_data_env2, node_noise_env1, 
                                     node_noise_env2, parents, node_idx, env1_test_result, 
                                     env2_test_result, save_dir):
        """
        Plot distance correlation independence test results.
        """
        plt.figure(figsize=(12, 8))

        # Environment 1 visualization
        plt.subplot(2, 2, 1)
        if len(parents) == 1:
            plt.scatter(parent_data_env1.flatten(), node_noise_env1, alpha=0.6, s=10)
            plt.xlabel(f'Parent X{parents[0]} values')
        else:
            plt.scatter(parent_data_env1[:, 0], node_noise_env1, alpha=0.6, s=10)
            plt.xlabel(f'Parent X{parents[0]} values (first parent)')
        plt.ylabel(f'Node X{node_idx} noise')
        plt.title(f'Env1: All Parents vs X{node_idx} noise\nDCor Score: {env1_test_result["dcor_score"]:.4f}')
        plt.grid(True, alpha=0.3)

        # Environment 1 DCor info
        plt.subplot(2, 2, 2)
        plt.text(0.1, 0.7, f'DCor Score: {env1_test_result["dcor_score"]:.4f}', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.text(0.1, 0.5, f'DCor p-value: {env1_test_result["dcor_p"]:.4f}', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.text(0.1, 0.3, f'Independent: {env1_test_result["is_independent"]}', 
                transform=plt.gca().transAxes, fontsize=14, weight='bold')
        plt.text(0.1, 0.1, f'Parents: {list(parents)}', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'Environment 1: DCor Test Results\nNode X{node_idx}')

        # Add color coding
        if env1_test_result['is_independent']:
            plt.gca().set_facecolor('#eeffee')  # Light green
        else:
            plt.gca().set_facecolor('#ffeeee')  # Light red

        # Environment 2 visualization
        plt.subplot(2, 2, 3)
        if len(parents) == 1:
            plt.scatter(parent_data_env2.flatten(), node_noise_env2, alpha=0.6, s=10, color='orange')
            plt.xlabel(f'Parent X{parents[0]} values')
        else:
            plt.scatter(parent_data_env2[:, 0], node_noise_env2, alpha=0.6, s=10, color='orange')
            plt.xlabel(f'Parent X{parents[0]} values (first parent)')
        plt.ylabel(f'Node X{node_idx} noise')
        plt.title(f'Env2: All Parents vs X{node_idx} noise\nDCor Score: {env2_test_result["dcor_score"]:.4f}')
        plt.grid(True, alpha=0.3)

        # Environment 2 DCor info
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.7, f'DCor Score: {env2_test_result["dcor_score"]:.4f}', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.text(0.1, 0.5, f'DCor p-value: {env2_test_result["dcor_p"]:.4f}', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.text(0.1, 0.3, f'Independent: {env2_test_result["is_independent"]}', 
                transform=plt.gca().transAxes, fontsize=14, weight='bold')
        plt.text(0.1, 0.05, f'Parents: {list(parents)}', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'Environment 2: DCor Test Results\nNode X{node_idx}')

        # Add color coding
        if env2_test_result['is_independent']:
            plt.gca().set_facecolor('#eeffee')  # Light green
        else:
            plt.gca().set_facecolor('#ffeeee')  # Light red

        plt.suptitle(f'Distance Correlation Independence Test: Node X{node_idx} vs All Parents')
        plt.tight_layout()

        # Save plot
        filename = f"dcor_independence_test_node_{node_idx}.png"
        filepath = os.path.join(save_dir, filename)
            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved DCor independence test plot: {filepath}")

        plt.show()
        plt.close()
    
    @staticmethod
    def plot_gam_fitting(standardization_result, node_idx, parents, save_dir):
        """
        Plot GAM fitting for conditional mean and conditional variance.
        """
        parent_data = standardization_result['parent_data']
        noise = standardization_result['noise']
        cond_mean = standardization_result['cond_mean']
        cond_std = standardization_result['cond_std']
        residuals = standardization_result['residuals']
        n_parents = standardization_result['n_parents']
        mean_r2 = standardization_result['mean_r2']
        var_r2 = standardization_result['var_r2']
        n_splines = standardization_result['n_splines']
        lam = standardization_result['lam']
        
        # Figure 1: Conditional Mean Fitting (GAM)
        n_cols = min(3, n_parents)
        n_rows = math.ceil(n_parents / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_parents == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_parents > 1 else axes
        
        for i, parent_idx in enumerate(parents):
            ax = axes[i]
            
            # Sort by parent values for visualization
            sort_idx = np.argsort(parent_data[:, i])
            parent_sorted = parent_data[sort_idx, i]
            noise_sorted = noise[sort_idx]
            cond_mean_sorted = cond_mean[sort_idx]
            
            # Scatter plot of actual data
            ax.scatter(parent_data[:, i], noise, alpha=0.2, s=5, color='gray', label='Actual noise')
            
            # Fitted conditional mean (GAM)
            ax.plot(parent_sorted, cond_mean_sorted, 'r-', linewidth=2, label=f'GAM mean (splines={n_splines})')
            
            ax.set_xlabel(f'Parent X{parent_idx}')
            ax.set_ylabel('Noise')
            ax.set_title(f'Conditional Mean (GAM): Parent X{parent_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_parents, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Conditional Mean (GAM): Node X{node_idx} [R²={mean_r2:.4f}]', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"gam_conditional_mean_fit_node_{node_idx}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved conditional mean fitting plot: {filepath}")
        plt.show()
        plt.close()
        
        # Figure 2: Conditional Variance (GAM)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_parents == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_parents > 1 else axes
        
        for i, parent_idx in enumerate(parents):
            ax = axes[i]
            
            # Sort by parent values
            sort_idx = np.argsort(parent_data[:, i])
            parent_sorted = parent_data[sort_idx, i]
            cond_std_sorted = cond_std[sort_idx]
            
            # Scatter plot of residuals
            ax.scatter(parent_data[:, i], residuals, alpha=0.2, s=5, 
                    color='gray', label='Residuals')
            
            # Conditional std bands (±1σ, ±2σ)
            ax.plot(parent_sorted, cond_std_sorted, 'r-', linewidth=2, label='GAM Std')
            ax.plot(parent_sorted, -cond_std_sorted, 'r-', linewidth=2)
            ax.fill_between(parent_sorted, -cond_std_sorted, cond_std_sorted,
                        alpha=0.2, color='red', label='±1σ band')
            ax.fill_between(parent_sorted, -2*cond_std_sorted, 2*cond_std_sorted,
                        alpha=0.1, color='blue', label='±2σ band')
            
            ax.set_xlabel(f'Parent X{parent_idx}')
            ax.set_ylabel('Residuals / GAM Std')
            ax.set_title(f'Conditional Variance (GAM): Parent X{parent_idx}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_parents, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Conditional Variance (GAM): Node X{node_idx} [R²={var_r2:.4f}]', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"gam_conditional_variance_fit_node_{node_idx}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved conditional variance fitting plot: {filepath}")
        plt.show()
        plt.close()
        
        # Figure 3: Combined view
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_parents == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_parents > 1 else axes
        
        for i, parent_idx in enumerate(parents):
            ax = axes[i]
            
            # Sort by parent values
            sort_idx = np.argsort(parent_data[:, i])
            parent_sorted = parent_data[sort_idx, i]
            cond_mean_sorted = cond_mean[sort_idx]
            cond_std_sorted = cond_std[sort_idx]
            
            # Scatter plot of actual data
            ax.scatter(parent_data[:, i], noise, alpha=0.2, s=5, color='gray', 
                    label='Actual noise')
            
            # Fitted conditional mean
            ax.plot(parent_sorted, cond_mean_sorted, 'r-', linewidth=2, 
                label=f'GAM mean (splines={n_splines})')
            
            # ±1σ and ±2σ bands
            ax.fill_between(parent_sorted, 
                        cond_mean_sorted - cond_std_sorted, 
                        cond_mean_sorted + cond_std_sorted,
                        alpha=0.3, color='red', label='±1σ band')
            ax.fill_between(parent_sorted,
                        cond_mean_sorted - 2*cond_std_sorted,
                        cond_mean_sorted + 2*cond_std_sorted,
                        alpha=0.1, color='blue', label='±2σ band')
            
            ax.set_xlabel(f'Parent X{parent_idx}')
            ax.set_ylabel('Noise')
            ax.set_title(f'Mean & Variance: Parent X{parent_idx}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_parents, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Conditional Mean & Variance Bands (GAM): Node X{node_idx}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"gam_mean_variance_combined_node_{node_idx}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved combined mean/variance plot: {filepath}")
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_gam_diagnostics(standardization_result, node_idx, parents, save_dir):
        """
        Plot GAM diagnostics including partial dependence plots and residuals.
        """
        parent_data = standardization_result['parent_data']
        residuals = standardization_result['residuals']
        n_parents = standardization_result['n_parents']
        gam_mean = standardization_result['gam_mean']
        n_splines = standardization_result['n_splines']
        parent_data_scaled = standardization_result['parent_data_scaled']
        
        # Figure: GAM Residual Diagnostics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Fitted
        fitted = gam_mean.predict(parent_data_scaled)
        ax = axes[0, 0]
        ax.scatter(fitted, residuals, alpha=0.3, s=5)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')
        ax.grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        ax = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)
        
        # 3. Histogram of residuals
        ax = axes[1, 0]
        ax.hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x_range, norm.pdf(x_range, np.mean(residuals), np.std(residuals)), 
            'r-', linewidth=2, label='Normal fit')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Histogram of Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Partial dependence plot for first parent
        ax = axes[1, 1]
        if n_parents > 0:
            # Generate partial dependence for first parent
            XX = gam_mean.generate_X_grid(term=0)
            pdep, confi = gam_mean.partial_dependence(term=0, X=XX, width=0.95)
            
            ax.plot(XX[:, 0], pdep, 'b-', linewidth=2, label='Partial Dependence')
            ax.fill_between(XX[:, 0], confi[:, 0], confi[:, 1], alpha=0.3, label='95% CI')
            ax.set_xlabel(f'Parent X{parents[0]}')
            ax.set_ylabel('Partial Effect')
            ax.set_title(f'GAM Partial Dependence: Parent X{parents[0]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
        
        plt.suptitle(f'GAM Diagnostics: Node X{node_idx}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"gam_residual_diagnostics_node_{node_idx}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved GAM residual diagnostics: {filepath}")
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_simultaneous_shift_test(standardization_result, chisq_test_result, 
                                     node_idx, shift_type, save_dir):
        """
        Visualize simultaneous shift test results for GAM-based approach.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        std_residuals = standardization_result['standardized_residuals']
        squared_residuals = chisq_test_result['squared_residuals']
        residuals = standardization_result['residuals']
        n_splines = standardization_result.get('n_splines', 'N/A')
        
        # 1. Original residuals distribution
        ax = axes[0, 0]
        ax.hist(residuals, bins=50, alpha=0.7, 
                density=True, color='blue', edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title(f'Node {node_idx}: Original Residuals')
        ax.grid(True, alpha=0.3)
        
        # 2. Standardized residuals vs Normal(0,1)
        ax = axes[0, 1]
        ax.hist(std_residuals, bins=50, alpha=0.7, density=True, 
                color='green', edgecolor='black', label='Standardized Residuals')
        x_range = np.linspace(std_residuals.min(), std_residuals.max(), 100)
        ax.plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2, label='N(0,1)')
        ax.set_xlabel('Standardized Residuals')
        ax.set_ylabel('Density')
        ax.set_title(f'Node {node_idx}: Standardized Residuals vs N(0,1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-Q plot for standardized residuals
        ax = axes[0, 2]
        stats.probplot(std_residuals, dist="norm", plot=ax)
        ax.set_title(f'Node {node_idx}: Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        # 4. Squared standardized residuals vs chi-square(1)
        ax = axes[1, 0]
        ax.hist(squared_residuals, bins=50, alpha=0.7, density=True, 
                color='orange', edgecolor='black', label='Squared Std. Residuals')
        x_range = np.linspace(0, np.percentile(squared_residuals, 99), 100)
        ax.plot(x_range, chi2.pdf(x_range, df=1), 'r-', linewidth=2, label='χ²(1)')
        ax.set_xlabel('Squared Standardized Residuals')
        ax.set_ylabel('Density')
        ax.set_title(f'Node {node_idx}: Squared Residuals vs χ²(1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. CDF comparison
        ax = axes[1, 1]
        sorted_data = np.sort(squared_residuals)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        theoretical_cdf = chi2.cdf(sorted_data, df=1)
        ax.plot(sorted_data, empirical_cdf, 'b-', label='Empirical CDF', linewidth=2)
        ax.plot(sorted_data, theoretical_cdf, 'r--', label='χ²(1) CDF', linewidth=2)
        ax.set_xlabel('Squared Standardized Residuals')
        ax.set_ylabel('CDF')
        ax.set_title(f'Node {node_idx}: CDF Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Test results summary
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Simultaneous Shift Test (GAM)
        ==============================
        Node: X{node_idx}
        n_splines: {n_splines}
        
        KS Statistic: {chisq_test_result['ks_statistic']:.4f}
        KS p-value: {chisq_test_result['ks_pvalue']:.4f}
        
        Similar to χ²(1): {chisq_test_result['is_similar_to_chisq']}
        
        Shift Type: {shift_type.upper()}
        
        Mean fit R²: {standardization_result['mean_r2']:.4f}
        Var fit R²: {standardization_result['var_r2']:.4f}
        
        Interpretation:
        {"✓ Function shift only" if shift_type == 'function_only' else "✓ Function + Noise shift"}
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='center', family='monospace')
        
        # Color code background
        if shift_type == 'function_only':
            ax.set_facecolor('#eeffee')  # Light green
        else:
            ax.set_facecolor('#ffeeee')  # Light red
        
        plt.suptitle(f'Simultaneous Shift Test (GAM): Node X{node_idx}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_dir is not None:
            filename = f"simultaneous_shift_test_node_{node_idx}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved simultaneous shift test plot: {filepath}")
        
        plt.show()
        plt.close()