import torch
import numpy as np
from scipy.stats import ks_2samp, kstest
import matplotlib.pyplot as plt
from scipy import stats
import os
import time

from fans import FANSAnalyzer


class FANSAblationAnalyzer:
    """
    Cross-environment noise transfer experiment.
    Verifies that CNF_env2.inv(CNF_env1(x_env1)) ~ x_env2
    when both flows are well-trained (z ~ N(0,I)).
    """

    def __init__(self, model_env1, model_env2, preparator, device,
                 scaler_env1, scaler_env2):
        self.analyzer_env1 = FANSAnalyzer(model_env1, preparator, device, scaler_env1,
                                          enable_visualizer=False)
        self.analyzer_env2 = FANSAnalyzer(model_env2, preparator, device, scaler_env2,
                                          enable_visualizer=False)
        self.device = device
        self.dag = preparator.adjacency(False).cpu().numpy()

    def cross_environment_generate(self, data_env1):
        """
        x_env1 -> CNF_env1 -> z_env1 -> CNF_env2.inv -> x_recon
        """
        z_env1 = self.analyzer_env1.x_to_z(data_env1)
        x_recon = self.analyzer_env2.z_to_x(z_env1)
        return x_recon, z_env1

    def verify(self, data_env1, data_env2, target_node, save_dir=None):
        """
        Verify that x_recon has the same distribution as x_env2 for target_node.

        Returns:
            result: dict with KS test statistics
        """
        data_env1 = data_env1.to(self.device)
        data_env2 = data_env2.to(self.device)

        x_recon, z_env1 = self.cross_environment_generate(data_env1)

        z_node = z_env1[:, target_node].cpu().numpy()
        recon_node = x_recon[:, target_node].cpu().numpy()
        env1_node = data_env1[:, target_node].cpu().numpy()
        env2_node = data_env2[:, target_node].cpu().numpy()

        ks_z, p_z = kstest(z_node, 'norm')
        ks_recon_env2, p_recon_env2 = ks_2samp(recon_node, env2_node)
        ks_env1_env2, p_env1_env2 = ks_2samp(env1_node, env2_node)

        result = {
            'target_node': target_node,
            'z_normality': {'ks_stat': float(ks_z), 'p_value': float(p_z)},
            'recon_vs_env2': {'ks_stat': float(ks_recon_env2), 'p_value': float(p_recon_env2)},
            'env1_vs_env2': {'ks_stat': float(ks_env1_env2), 'p_value': float(p_env1_env2)},
        }

        print(f"\n=== Ablation Verification for Node {target_node} ===")
        print(f"  z ~ N(0,1)?        KS={ks_z:.4f}, p={p_z:.4f}")
        print(f"  x_recon ~ x_env2?  KS={ks_recon_env2:.4f}, p={p_recon_env2:.4f}")
        print(f"  x_env1 ~ x_env2?   KS={ks_env1_env2:.4f}, p={p_env1_env2:.4f}")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            self._plot_verification(
                z_node, recon_node, env1_node, env2_node,
                target_node, result, save_dir
            )

        return result

    def _plot_verification(self, z_node, recon_node, env1_node, env2_node,
                           target_node, result, save_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(z_node, bins=50, density=True, alpha=0.7, label='z_env1')
        x_range = np.linspace(-4, 4, 200)
        axes[0].plot(x_range, stats.norm.pdf(x_range), 'r-', lw=2, label='N(0,1)')
        axes[0].set_title(
            f'Node {target_node}: z normality '
            f'(p={result["z_normality"]["p_value"]:.4f})')
        axes[0].legend()
        axes[0].set_xlabel('z')
        axes[0].set_ylabel('Density')

        axes[1].hist(env1_node, bins=50, density=True, alpha=0.5, label='x_env1')
        axes[1].hist(env2_node, bins=50, density=True, alpha=0.5, label='x_env2')
        axes[1].hist(recon_node, bins=50, density=True, alpha=0.5, label='x_recon')
        axes[1].set_title(
            f'Node {target_node}: recon vs env2 '
            f'(KS p={result["recon_vs_env2"]["p_value"]:.4f})')
        axes[1].legend()
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Density')

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f'ablation_node_{target_node}.png'),
            dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved to: {save_dir}/ablation_node_{target_node}.png")
