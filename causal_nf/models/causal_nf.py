import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tueplots import bundles
import causal_nf.utils.io as causal_io
from causal_nf.models.base_model import BaseLightning
from causal_nf.utils.graph import ancestor_matrix
from causal_nf.utils.optimizers import build_optimizer, build_scheduler
from causal_nf.utils.pairwise.mmd import maximum_mean_discrepancy
from causal_nf.modules.causal_nf import CausalNormalizingFlow

import numpy as np

class CausalNFightning(BaseLightning):
    def __init__(
        self,
        preparator,
        model: CausalNormalizingFlow,
        init_fn=None,
        plot=True,
        regularize=False,
        kl="forward",
    ):
        super(CausalNFightning, self).__init__(preparator, init_fn=init_fn)
        self.adjacency_diag = preparator.adjacency(True)
        self.adjacency_no_diag = preparator.adjacency(False)
        if not isinstance(self.adjacency_diag, torch.Tensor):
            self.adjacency_diag = torch.tensor(self.adjacency_diag, dtype=torch.float32)
        if not isinstance(self.adjacency_no_diag, torch.Tensor):
            self.adjacency_no_diag = torch.tensor(self.adjacency_no_diag, dtype=torch.float32)

        self.model = model
        self.plot = plot
        self.regularize = regularize
        self.kl = kl
        self.set_input_scaler()
        self.reset_parameters()

    def to(self, device):
        """Move only model and scaler_transform to GPU"""
        # Call parent class's to() method (move model to GPU)
        result = super().to(device)
            
        # Move to GPU only once and check device
        if hasattr(self, 'adjacency_diag') and self.adjacency_diag is not None:
            if self.adjacency_diag.device != device:
                self.adjacency_diag = self.adjacency_diag.to(device)
        
        if hasattr(self, 'adjacency_no_diag') and self.adjacency_no_diag is not None:
            if self.adjacency_no_diag.device != device:
                self.adjacency_no_diag = self.adjacency_no_diag.to(device)

        # Move scaler_transform to GPU (only once)
        if hasattr(self.preparator, 'scaler_transform') and self.preparator.scaler_transform is not None:
            if not hasattr(self.preparator.scaler_transform, 'device') or self.preparator.scaler_transform.device != device:
                self.preparator.scaler_transform = self.preparator.scaler_transform.to(device)
                print(f"Moved scaler_transform to {device}")
        
        return result

    def reset_parameters(self):
        super(CausalNFightning, self).reset_parameters()

    def set_input_scaler(self):
        self.input_scaler = self.preparator.get_scaler(fit=True)
        print(self.input_scaler)
        self.model.set_adjacency(self.preparator.adjacency())
    def get_x_norm(self, batch, batch_size=None):
        x_norm = self.input_scaler.transform(batch[0].to(self.device), inplace=False)
        # Replace NaN values with 0
        x_norm = torch.where(torch.isnan(x_norm), torch.zeros_like(x_norm), x_norm)
        return x_norm
    def forward(self, batch, **kwargs):
        x_norm = self.get_x_norm(batch=batch)

        tic = time.time()
        output = self.model(x_norm)

        if self.regularize:
            jac = torch.autograd.functional.jacobian(
                self.model.flow().transform, x_norm.mean(0), create_graph=True
            )
            adj = self.adjacency_diag
            loss_ = torch.norm(jac[(adj == 0.0)], p=2)
            output["loss"] = output["loss"] + loss_
        output["time_forward"] = self.compute_time(tic, x_norm.shape[0])
        return output

    def compute_time(self, tic, num_samples):
        delta_time = (time.time() - tic) * 1000
        return torch.tensor(delta_time / num_samples * 1000)

    @torch.no_grad()
    def predict(
        self,
        batch,
        observational=False,
        intervene=False,
        counterfactual=False,
        ate=False,
    ):
        output = {}
        x = batch[0]
        
        # Prevent duplicate GPU transfers
        if hasattr(x, 'device') and x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        elif not hasattr(x, 'device'):
            x = x.to(self.device)        
        
        n = x.shape[0]

        if self.preparator.external_data is None:
            with torch.enable_grad():
                x_cpu = x.cpu()
                log_prob_true = self.preparator.log_prob(x_cpu)
                output["log_prob_true"] = log_prob_true.to(self.device)

        tic = time.time()
        log_prob = self.model.log_prob(x, scaler=self.preparator.scaler_transform)
        output["time_log_prob"] = self.compute_time(tic, n)
        output["loss"] = -log_prob
        output["log_prob"] = log_prob

        if self.preparator.external_data is not None:
            observational = False
            intervene = False
            counterfactual = False
            ate = False
            print('External data mode: Skipping true log probability calculations')

        observational = False
        intervene = False
        counterfactual = False
        ate = False

        if observational:
            tic = time.time()
            obs_dict = self.model.sample((n,))
            output["time_sample_obs"] = self.compute_time(tic, n)
            x_obs_norm = obs_dict["x_obs"]
            x_obs = self.input_scaler.inverse_transform(x_obs_norm, inplace=False)
            if self.plot:
                output["x"] = self.preparator.post_process(x)
            if self.plot:
                output["x_obs"] = self.preparator.post_process(x_obs)
            mmd_value = maximum_mean_discrepancy(x, x_obs, sigma=None)
            output[f"mmd_obs"] = mmd_value

            with torch.enable_grad():
                x_obs_cpu = x_obs.cpu()
                x_cpu = x.cpu()
                log_p_with_x_sample = self.preparator.log_prob(x_obs_cpu)
                log_p_with_x = self.preparator.log_prob(x_cpu)
            output["log_prob_p"] = log_p_with_x_sample
            log_q_with_x_sample = self.model.log_prob(
                x_obs, scaler=self.preparator.scaler_transform
            )

            log_prob_cpu = log_prob.cpu()
            log_q_with_x_sample_cpu = log_q_with_x_sample.cpu()
            kl_distance = (
                log_p_with_x + log_q_with_x_sample_cpu - log_p_with_x_sample - log_prob_cpu
            )
            output["kl_distance"] = kl_distance

        if intervene:
            intervention_list = self.preparator.get_intervention_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                value = int_dict["value"]
                index = int_dict["index"]
                tic = time.time()
                x_int = self.model.intervene(
                    index=index,
                    value=value,
                    shape=(n,),
                    scaler=self.preparator.scaler_transform,
                )
                delta_times.append(self.compute_time(tic, n))

                if self.plot:
                    output[f"x_int_{index + 1}={name}"] = self.preparator.post_process(
                        x_int
                    )

                x_int_true = self.preparator.intervene(
                    index=index, value=value, shape=(n,)
                )
                if self.plot:
                    output[
                        f"x_int_{index + 1}={name}_true"
                    ] = self.preparator.post_process(x_int_true)

                
                x_int_cpu = x_int.cpu()
                mmd_value = maximum_mean_discrepancy(x_int_cpu, x_int_true, sigma=None)
                output[f"mmd_int_x{index + 1}={name}"] = mmd_value

            delta_time = torch.stack(delta_times).mean()
            output["time_intervene"] = delta_time
        if counterfactual:
            intervention_list = self.preparator.get_intervention_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                value = int_dict["value"]
                index = int_dict["index"]
                tic = time.time()
                x_cf = self.model.compute_counterfactual(
                    x_factual=x,
                    index=index,
                    value=value,
                    scaler=self.preparator.scaler_transform,
                )
                delta_times.append(self.compute_time(tic, n))

                x_cf_true = self.preparator.compute_counterfactual(x, index, value)

                diff_cf = x_cf_true - x_cf

                rmse = torch.sqrt((diff_cf**2).sum(1))
                output[f"rmse_cf_x{index + 1}={name}"] = rmse
                mae = diff_cf.abs().sum(1)
                output[f"mse_cf_x{index + 1}={name}"] = mae

            delta_time = torch.stack(delta_times).mean()
            output["time_cf"] = delta_time

        if ate:
            intervention_list = self.preparator.get_ate_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                a = int_dict["a"]
                b = int_dict["b"]
                index = int_dict["index"]
                tic = time.time()
                ate = self.model.compute_ate(
                    index,
                    a=a,
                    b=b,
                    num_samples=10000,
                    scaler=self.preparator.scaler_transform,
                )
                delta_times.append(self.compute_time(tic, 10000))

                ate_true = self.preparator.compute_ate(
                    index, a=a, b=b, num_samples=10000
                )
                ate_cpu = ate.cpu()
                diff_ate = ate_true - ate_cpu

                rmse = torch.sqrt((diff_ate**2).sum())
                output[f"rmse_ate_x{index + 1}={name}"] = rmse

            delta_time = torch.stack(delta_times).mean()
            output["time_ate"] = delta_time

        return output

    def vi(self, n_samples):

        flow = self.model.flow()
        z = flow.base.rsample((n_samples,))
        x_norm = flow.transform.inv(z)
        x = self.input_scaler.inverse_transform(x_norm, inplace=False)
        cte = min(1.0, self.current_epoch / 1000)
        output = self.model.vi(x, self.preparator.scm, cte)

        if self.regularize:
            jac = torch.autograd.functional.jacobian(
                flow.transform.inv, z.mean(0), create_graph=True
            )
            adj = self.adjacency_diag
            adj = ancestor_matrix(adj)

            loss_ = torch.norm(jac[(adj == 0.0)], p=2)
            output["loss"] = output["loss"] + loss_

        return output

    # process inside the training loop
    def training_step(self, train_batch, batch_idx):

        if self.kl == "forward":
            loss_dict = self(train_batch)
        elif self.kl == "backward":
            loss_dict = self.vi(train_batch[0].shape[0])
        else:
            raise AttributeError

        loss_dict["loss"] = loss_dict["loss"].mean()
        log_dict = {}
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            # Existing jacobian loss
            jacobian_output = self.jacobian_losses(train_batch)
            loss_dict.update(jacobian_output)
            
            # Add independence loss (based on Distance Correlation)
            independence_output = self.independence_losses(
                train_batch, 
                weight_z_independence=1,  # Weight for z-z correlation constraint
                weight_parent_z_independence=1  # Weight for parent-z independence constraint
            )

            print(independence_output)
            # Add independence constraint to total loss
            for key, value in independence_output.items():
                loss_dict["loss"] = loss_dict["loss"] + value
                loss_dict[key] = value

        self.update_log_dict(log_dict=log_dict, my_dict=loss_dict, regex=r"^(?!x_).*$")
        return loss_dict

    def validation_step(self, batch, batch_idx):
        self.eval()

        if self.current_epoch % 10 == 1:
            observational = batch_idx == 0
            intervene = False
            ate = False
        else:
            observational = False
            intervene = False
            ate = False

        loss_dict = self.predict(
            batch,
            observational=observational,
            intervene=intervene,
            counterfactual=False,
            ate=ate,
        )

        log_dict = {}

        self.update_log_dict(
            log_dict=log_dict, my_dict=loss_dict, regex=r"^(?!.*x_).*$"
        )

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            output = self.jacobian_losses(batch)
            log_dict.update(output)

        return log_dict

    def add_noise(self, x):
        # Calculate the standard deviation of each column
        std = torch.std(x, dim=0).mul(100).round() / 100.0
        # Find the columns that are constant (i.e., have a standard deviation of 0)
        constant_mask = std == 0
        # # Generate a small amount of noise for each constant column
        # noise = torch.rand(x.shape[0], sum(constant_mask)) * 2.0 - 1.0
        noise = torch.randn(x.shape[0], sum(constant_mask))
        # Add the noise to the corresponding columns
        x[:, constant_mask] += noise * 0.01
        return x

    def compute_metrics_stats(self, outputs):

        metric_stats = super(CausalNFightning, self).compute_metrics_stats(outputs)

        metric_stats = {
            key: value for key, value in metric_stats.items() if "x_" not in key
        }

        data = {}

        plot_intervene = False

        for output_i in outputs:
            for key, values in output_i.items():
                if "x" in key:
                    if key not in data:
                        data[key] = []
                    data[key].append(values)

                    if "x_int" in key:
                        plot_intervene = True

        n = 256
        split = self.preparator.current_split
        filename = os.path.join(self.logger.save_dir, f"split={split}_name=")
        if "x_obs" in data and split != "train":
            x_obs = data["x_obs"]
            x = data["x"]
            x_obs = torch.cat(x_obs, dim=0)[:n]
            x = torch.cat(x, dim=0)[:n]
            df = self.preparator.create_df([x, x_obs], ["real", "fake"])

            fig = self.preparator._plot_data(df=df, hue="mode")
            try:
                wandb.log({"x_obs": wandb.Image(fig)}, step=self.current_epoch)
            except:
                fig.savefig(f"{filename}x_obs.pdf")
            plt.close("all")

        if plot_intervene and split != "train":
            for key in data:
                if "x_int" in key and "true" not in key:
                    x_int = data[key]
                    x_int_true = data[key + "_true"]
                    x_int = torch.cat(x_int, dim=0)[:n]
                    x_int_true = torch.cat(x_int_true, dim=0)[:n]

                    x_int = self.add_noise(x_int)
                    x_int_true = self.add_noise(x_int_true)

                    df = self.preparator.create_df(
                        [x_int_true, x_int], ["real", "fake"]
                    )
                    fig = self.preparator._plot_data(df=df, hue="mode")
                    try:
                        wandb.log({key: wandb.Image(fig)}, step=self.current_epoch)
                    except:
                        fig.savefig(f"{filename}{key}.pdf")

                    plt.close("all")
        return metric_stats

    def test_step(self, batch, batch_idx):

        self.eval()

        observational = batch_idx < 1
        # observational = False
        intervene = batch_idx < 1
        # intervene = False
        counterfactual = batch_idx < 1
        ate = batch_idx < 1

        loss_dict = self.predict(
            batch,
            observational=observational,
            intervene=intervene,
            counterfactual=counterfactual,
            ate=ate,
        )

        log_dict = {}

        self.update_log_dict(log_dict=log_dict, my_dict=loss_dict)
        split = self.preparator.current_split
        return log_dict

    def plot(self):
        raise NotImplementedError

    def _plot_jacobian(self, J, title="Jacobian Matrix", variable="x"):
        if isinstance(J, torch.Tensor):
            J = J.detach().numpy()

        J_abs = np.absolute(J)
        # Create a figure and axis object
        fig, ax = plt.subplots()

        # Plot the matrix using the axis object's `matshow` function
        height, width = J.shape
        fig_aspect_ratio = fig.get_figheight() / fig.get_figwidth()
        data_aspect_ratio = (height / width) * fig_aspect_ratio
        # Plot the matrix using the axis object's `matshow` function
        cax = ax.matshow(
            J_abs, aspect=data_aspect_ratio, cmap="viridis"
        )  # You can change the colormap to your preference

        # Add a colorbar to the plot for easy interpretation
        fig.colorbar(cax)

        # Set the title for the axis object
        ax.set_title(f"{title} {variable}")

        # Label the x and y axes
        ax.set_xticks(range(J.shape[1]))
        ax.set_yticks(range(J.shape[0]))

        xticks = [
            "$\\frac{{ \\partial f_m }}{{ \\partial {}_{} }}$".format(variable, i)
            for i in range(1, J.shape[1] + 1)
        ]
        ax.set_xticklabels(xticks)
        yticks = [
            "$\\frac{{ \\partial f_{} }}{{ \\partial {}_n }}$".format(i, variable)
            for i in range(1, J.shape[1] + 1)
        ]
        ax.set_yticklabels(yticks)

        # Display the values of the Jacobian matrix with 2 decimal points
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                value = J[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")

        return fig
    
    def configure_optimizers(self):
        self.lr = self.optim_config.base_lr
        causal_io.print_debug(f"Setting lr: {self.lr}")

        params = self.model.parameters()
        opt = build_optimizer(optim_config=self.optim_config, params=params)

        output = {}

        if isinstance(self.optim_config.scheduler, str):
            sched = build_scheduler(optim_config=self.optim_config, optimizer=opt)
            output["optimizer"] = opt
            output["lr_scheduler"] = sched
            output["monitor"] = "loss"  # Previously used val_loss
        else:
            output["optimizer"] = opt
        return output

    def independence_losses(self, batch, weight_z_independence=10.0, weight_parent_z_independence=10.0):
        """
        Method to compute independence constraints (based on Distance Correlation)
        """
        output = {}
        x_norm = self.get_x_norm(batch=batch)
        
        # Transform x → z
        with torch.enable_grad():
            z = self.model.flow().transform(x_norm)
        
        # 1. Independence constraint among Z (correlation-based)
        z_independence_loss = self.compute_z_correlation_loss(z)
        output["loss_z_independence"] = z_independence_loss * weight_z_independence

        # 2. Independence constraint between parent variables and z (using Distance correlation)
        parent_z_independence_loss = self.compute_parent_z_dcor_loss(x_norm, z)
        output["loss_parent_z_independence"] = parent_z_independence_loss * weight_parent_z_independence
        
        print(f"[Epoch {self.current_epoch}] Z Correlation Loss: {z_independence_loss:.10e}")
        print(f"[Epoch {self.current_epoch}] Parent-Z DCOR Loss: {parent_z_independence_loss:.10e}")
        
        return output

    def compute_z_correlation_loss(self, z):
        """
        Compute independence loss among z (correlation-based)
        """
        z_centered = z - z.mean(dim=0, keepdim=True)
        z_std = z_centered.std(dim=0, keepdim=True) + 1e-8
        z_normalized = z_centered / z_std
        
        cov_matrix = torch.mm(z_normalized.t(), z_normalized) / (z.shape[0] - 1)
        
        off_diagonal_mask = ~torch.eye(z.shape[1], dtype=torch.bool, device=z.device)
        correlation_loss = (cov_matrix[off_diagonal_mask] ** 2).mean()
        
        return correlation_loss

    def compute_distance_correlation_differentiable(self, x, y):
        """
        Differentiable distance correlation
        """
        n = x.shape[0]
        print(f"[Distance Correlation] Batch size: {n}, X shape: {x.shape}, Y shape: {y.shape}")

        if x.dim() == 1: x = x.unsqueeze(1)
        if y.dim() == 1: y = y.unsqueeze(1)
            
        # Compute pairwise distances
        x_expanded = x.unsqueeze(1)
        x_distances = torch.norm(x_expanded - x.unsqueeze(0), dim=2)
        
        y_expanded = y.unsqueeze(1)
        y_distances = torch.norm(y_expanded - y.unsqueeze(0), dim=2)
        
        # Double centering
        def double_center(dist_matrix):
            row_means = dist_matrix.mean(dim=1, keepdim=True)
            col_means = dist_matrix.mean(dim=0, keepdim=True)
            grand_mean = dist_matrix.mean()
            centered = dist_matrix - row_means - col_means + grand_mean
            return centered
        
        a_matrix = double_center(x_distances)
        b_matrix = double_center(y_distances)
        
        # Compute dCov^2(X,Y)
        dcov_xy_squared = (a_matrix * b_matrix).sum() / (n * n)
        
        # Compute dCov^2(X,X) and dCov^2(Y,Y)
        dcov_xx_squared = (a_matrix * a_matrix).sum() / (n * n)
        dcov_yy_squared = (b_matrix * b_matrix).sum() / (n * n)
        
        # Distance correlation
        denominator = torch.sqrt(dcov_xx_squared * dcov_yy_squared + 1e-10)
        dcor = dcov_xy_squared / denominator
        return dcor

    def compute_parent_z_dcor_loss(self, x, z):
        """
        Independence loss based on Distance correlation
        Correctly find parent nodes from upper triangular matrix
        If adj[i,j] = 1, then j → i (j is parent, i is child)
        """
        print(f"[Parent-Z DCOR] Input batch size: {x.shape[0]}, Num variables: {x.shape[1]}")
        total_dcor_loss = torch.tensor(0.0, device=x.device)
        num_constraints = 0
        for i in range(x.shape[1]):  # i is the child node
            parents = torch.where(self.adjacency_no_diag[i, :] == 1)[0]
            
            if len(parents) > 0:
                x_parents = x[:, parents]  # (batch_size, num_parents)
                z_i = z[:, i:i+1]          # (batch_size, 1)
                print(f"  Node {i}: Parents {parents.tolist()}, X_parents shape: {x_parents.shape}, Z_i shape: {z_i.shape}")
                dcor_value = self.compute_distance_correlation_differentiable(x_parents, z_i)
                total_dcor_loss = total_dcor_loss + dcor_value
                num_constraints += 1
        
        if num_constraints > 0:
            return total_dcor_loss / num_constraints
        else:
            return torch.tensor(0.0, device=x.device)

    def jacobian_losses(self, batch, filename=None):
        output = {}
        x_norm = self.get_x_norm(batch=batch)
        jac_x = self.model.compute_jacobian(x=x_norm)[-1]
        print(f"  jac_x:\n{jac_x}")
        
        adj = self.adjacency_diag
        triangular = torch.tril(torch.ones(adj.shape, device=self.device), diagonal=-1).bool()

        mask = (adj == 0.0) * triangular
        print(f"  adj:\n{adj}")
        print(f"  triangular:\n{triangular}")
        loss_ = torch.absolute(jac_x[mask]).mean()
        print(f"[Epoch {self.current_epoch}] Jacobian X Loss: {loss_:.10e}")
        output["loss_jacobian_x"] = loss_
        
        return output