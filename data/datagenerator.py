import random, os
import numpy as np
import igraph as ig
import torch
import torch.distributions as distr
from torch.distributions import Independent, Normal, Laplace
from heterogeneous import Heterogeneous
from typing import Union, List, Tuple, Dict, Optional
from scipy import stats

# The code below is from Chen, Tianyu, et al. "iSCAN: identifying causal mechanism shifts among nonlinear additive noise models." Advances in Neural Information Processing Systems 36 (2023): 44671-44706.

def set_seed(seed: int = 42) -> None:
    """
    Sets random seed of ``random`` and ``numpy`` to specified value
    for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Value for RNG, by default 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def softplus(x):
    """
    Softplus activation function: log(1 + exp(x))
    """
    return np.log(1 + np.exp(x))

class DataGenerator:
    """
    Generate synthetic data to evaluate shifted nodes and shifted edges
    """
    # New 6 shift cases
    SHIFT_CASE1 = "sin_cos_noise_N02"  # sin->cos, N(0,2)
    SHIFT_CASE2 = "sin_cos_noise_laplace"  # sin->cos, Laplace(0,1)
    SHIFT_CASE3 = "pa_delete_noise_N02"  # PA_j delete, N(0,2)
    SHIFT_CASE4 = "pa_delete_noise_laplace"  # PA_j delete, Laplace(0,1)
    SHIFT_CASE5 = "sin_cos_pa_delete"  # sin->cos, PA_j delete
    SHIFT_CASE6 = "noise_N02_laplace"  # N(0,2), Laplace(0,1)
    
    def __init__(self, d: int, s0: int, graph_type: str, 
                 shift_case: str = SHIFT_CASE1
               ):
        """
        Defines the class of graphs and data to be generated.

        Parameters
        ----------
        d : int
            Number of variables
        s0 : int
            Expected number of edges in the random graphs
        graph_type : str
            One of ``["ER", "SF"]``. ``ER`` and ``SF`` refer to Erdos-Renyi and Scale-free graphs, respectively.
        shift_case : str, optional
            Type of shift to apply to nodes. One of the six predefined cases.
        """
        self.d, self.s0, self.graph_type = d, s0, graph_type
        self.shift_case = shift_case
        
        # Initialize node shift lists
        self.sin_cos_shift_nodes = []  # nodes with sin->cos function shift
        self.pa_delete_shift_nodes = []  # nodes with parent deletion
        self.noise_n02_shift_nodes = []  # nodes with N(0,2) noise
        self.noise_laplace_shift_nodes = []  # nodes with Laplace(0,1) noise
        self.deleted_parents = {}  # dict storing deleted parents for each node
    
    def _create_multivariate_normal_distribution(self) -> Independent:
        """
        Create multivariate normal distribution N(0,1) for base environment.
        
        Returns:
            Independent: Multivariate normal distribution
        """
        return Independent(
            Normal(
                torch.zeros(self.d),
                torch.ones(self.d)  # N(0,1)
            ),
            1
        )
    
    def _create_heterogeneous_distribution(self, shifted_nodes: np.ndarray) -> Independent:
        """
        Create heterogeneous distribution with different noise types for shifted nodes.
        
        Args:
            shifted_nodes: Array of shifted node indices
            
        Returns:
            Independent: Heterogeneous multivariate distribution
        """
        distr_list = []
        
        for i in range(self.d):
            if i in self.noise_laplace_shift_nodes:
                # Laplace distribution for Laplace noise shifted nodes
                dist = Laplace(
                    torch.tensor([0.0]), 
                    torch.tensor([1.0])  # Laplace(0,1)
                )
            elif i in self.noise_n02_shift_nodes:
                # Normal distribution with higher variance for N(0,2) shifted nodes
                dist = Normal(
                    torch.tensor([0.0]), 
                    torch.tensor([np.sqrt(2.0)])  # N(0,2) -> std = sqrt(2)
                )
            else:
                # Standard normal distribution N(0,1)
                dist = Normal(
                    torch.tensor([0.0]), 
                    torch.tensor([1.0])
                )
            distr_list.append(dist)
        
        return Independent(Heterogeneous(distr_list), 1)
    
    def _sample_multivariate_noise(self, distribution: Independent, n_samples: int) -> np.ndarray:
        """
        Sample noise from multivariate distribution.
        
        Args:
            distribution: Multivariate distribution
            n_samples: Number of samples
            
        Returns:
            np.ndarray: Noise samples of shape (n_samples, d)
        """
        with torch.no_grad():
            samples = distribution.sample((n_samples,))
        return samples.numpy()

    def _simulate_dag(self, d: int, s0: int, graph_type: str) -> np.ndarray:
        """
        Simulate random DAG with some expected number of edges.

        Parameters
        ----------
        d : int
            num of nodes
        s0 : int
            expected num of edges
        graph_type : str
            ER, SF

        Returns
        -------
        np.ndarray
            :math:`(d, d)` binary adj matrix of a sampled DAG
        """
        def _random_acyclic_orientation(B_und):
            return np.triu(B_und, k=1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        if graph_type == 'ER': # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF': # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
            B_und = _graph_to_adjmat(G)
            B = _random_acyclic_orientation(B_und)
        else:
            raise ValueError('unknown graph type')
        
        assert ig.Graph.Adjacency(B.tolist()).is_dag()
        
        return B

    def _choose_shifted_nodes(self, adj: np.ndarray, num_shifted_nodes: int) -> np.ndarray:
        """
        Randomly choose shifted nodes from non-root nodes

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix
        num_shifted_nodes : int
            Number of desired shifted nodes

        Returns
        -------
        np.ndarray
            Shifted nodes
        """
        roots = np.where(adj.sum(axis=0) == 0)[0]
        non_roots = np.setdiff1d(list(range(self.d)), roots)
        
        # uniformly choose shifted nodes from non-root nodes
        if len(non_roots) > 0:
            shifted_nodes = np.random.choice(non_roots, min(num_shifted_nodes, len(non_roots)), replace=False)
        else:
            shifted_nodes = np.array([], dtype=int)
        
        return shifted_nodes
    
    def _assign_shift_types(self, shifted_nodes: np.ndarray) -> None:
        """
        Assign shift types to nodes based on the specified shift case.
        
        Parameters
        ----------
        shifted_nodes : np.ndarray
            Array of shifted node indices
        """
        # Reset shift type lists
        self.sin_cos_shift_nodes = []
        self.pa_delete_shift_nodes = []
        self.noise_n02_shift_nodes = []
        self.noise_laplace_shift_nodes = []
        self.deleted_parents = {}
        
        if len(shifted_nodes) == 0:
            return
        
        # For cases with 2 nodes: split evenly
        # For cases with 4 nodes: apply same type twice
        n_half = len(shifted_nodes) // 2
        first_half = shifted_nodes[:n_half]
        second_half = shifted_nodes[n_half:]
        
        # Assign shift types based on case
        if self.shift_case == self.SHIFT_CASE1:  # sin->cos, N(0,2)
            self.sin_cos_shift_nodes = list(first_half)
            self.noise_n02_shift_nodes = list(second_half)
        elif self.shift_case == self.SHIFT_CASE2:  # sin->cos, Laplace(0,1)
            self.sin_cos_shift_nodes = list(first_half)
            self.noise_laplace_shift_nodes = list(second_half)
        elif self.shift_case == self.SHIFT_CASE3:  # PA_j delete, N(0,2)
            self.pa_delete_shift_nodes = list(first_half)
            self.noise_n02_shift_nodes = list(second_half)
        elif self.shift_case == self.SHIFT_CASE4:  # PA_j delete, Laplace(0,1)
            self.pa_delete_shift_nodes = list(first_half)
            self.noise_laplace_shift_nodes = list(second_half)
        elif self.shift_case == self.SHIFT_CASE5:  # sin->cos, PA_j delete
            self.sin_cos_shift_nodes = list(first_half)
            self.pa_delete_shift_nodes = list(second_half)
        elif self.shift_case == self.SHIFT_CASE6:  # N(0,2), Laplace(0,1)
            self.noise_n02_shift_nodes = list(first_half)
            self.noise_laplace_shift_nodes = list(second_half)
        else:
            raise ValueError(f"Unknown shift case: {self.shift_case}")
    
    def _prepare_parent_deletions(self, adj: np.ndarray) -> None:
        """
        Prepare parent deletions for nodes that need PA_j deletion.
        """
        for node in self.pa_delete_shift_nodes:
            parents = np.nonzero(adj[:, node])[0]
            if len(parents) > 0:
                # Randomly choose one parent to delete
                deleted_parent = np.random.choice(parents)
                self.deleted_parents[int(node)] = int(deleted_parent)
    
    def _sample_from_same_structs(self, adj: np.ndarray, 
                                  shifted_nodes: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data with the same DAG structure for both environments using new formula
        X_j = Σsin(PA_j²) + exp(0.5 * ΣPA_j) * N_j
        
        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix
        shifted_nodes : np.ndarray
            Set of shifted nodes

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Datasets X and Y
        """
        # Assign shift types based on the specified case
        self._assign_shift_types(shifted_nodes)
        
        # Prepare parent deletions for PA_j deletion nodes
        self._prepare_parent_deletions(adj)
        
        # Create multivariate distributions
        normal_dist = self._create_multivariate_normal_distribution()
        heterogeneous_dist = self._create_heterogeneous_distribution(shifted_nodes)
        
        # Sample noise for both environments
        print(f"Sampling noise for Environment 1 (Normal): shape=({self.n}, {self.d})")
        noise_env1 = self._sample_multivariate_noise(normal_dist, self.n)
        
        print(f"Sampling noise for Environment 2 (Heterogeneous): shape=({self.n}, {self.d})")
        noise_env2 = self._sample_multivariate_noise(heterogeneous_dist, self.n)
        
        # Initialize data arrays
        data_env1 = np.zeros((self.n, self.d))
        data_env2 = np.zeros((self.n, self.d))
        
        # Copy noise initially
        data_env1[:] = noise_env1[:]
        data_env2[:] = noise_env2[:]
        
        print(f"Shifted nodes: {shifted_nodes}")
        print(f"Sin->Cos shift nodes: {self.sin_cos_shift_nodes}")
        print(f"PA delete shift nodes: {self.pa_delete_shift_nodes}")
        print(f"N(0,2) noise shift nodes: {self.noise_n02_shift_nodes}")
        print(f"Laplace noise shift nodes: {self.noise_laplace_shift_nodes}")
        print(f"Deleted parents: {self.deleted_parents}")
        
        # Apply structural equations with new formula
        # X_j = Σsin(PA_j²) + exp(0.5 * ΣPA_j) * N_j
        for i in range(self.d):
            parents = np.nonzero(adj[:, i])[0]
            
            if len(parents) > 0:
                # For dataset X (original environment)
                sum_sin_parents = np.zeros(self.n)
                sum_parents = np.zeros(self.n)
                
                for j in parents:
                    sum_sin_parents += np.sin(data_env1[:, j] ** 2)
                    sum_parents += data_env1[:, j]
                
                # X_j = Σsin(PA_j²) + exp(0.5 * ΣPA_j) * N_j
                data_env1[:, i] = sum_sin_parents + 1/(1+np.exp(-1 * sum_parents)) * noise_env1[:, i]
                
                # For dataset Y (environment with shifts)
                # Get parents for this node (potentially modified)
                env2_parents = parents.copy()
                if i in self.pa_delete_shift_nodes and i in self.deleted_parents:
                    # Remove the deleted parent
                    deleted_parent = self.deleted_parents[i]
                    env2_parents = env2_parents[env2_parents != deleted_parent]
                
                if len(env2_parents) > 0:
                    sum_parents_y = np.zeros(self.n)
                    
                    if i in self.sin_cos_shift_nodes:
                        # Function shift: sin(PA_j²) -> 4cos(2PA_j² - 3PA_j)
                        sum_func_parents = np.zeros(self.n)
                        for j in env2_parents:
                            sum_func_parents += np.cos(2 * data_env2[:, j]**2 - 3 * data_env2[:, j])
                            sum_parents_y += data_env2[:, j]
                    else:
                        # No function shift: use original sin formula
                        sum_func_parents = np.zeros(self.n)
                        for j in env2_parents:
                            sum_func_parents += np.sin(data_env2[:, j] ** 2)
                            sum_parents_y += data_env2[:, j]
                    
                    data_env2[:, i] = sum_func_parents + 1/(1+np.exp(-1 * sum_parents_y)) * noise_env2[:, i]
                else:
                    # No parents after deletion: just noise with exponential scaling
                    data_env2[:, i] = noise_env2[:, i]
            else:
                # No parents: just assign noise (no exponential scaling)
                data_env1[:, i] = noise_env1[:, i]
                data_env2[:, i] = noise_env2[:, i]
        
        return data_env1, data_env2

    def sample(self, n: int, 
               num_shifted_nodes: int, 
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples two datasets from randomly generated DAGs

        Parameters
        ----------
        n : int
            Number of samples
        num_shifted_nodes : int
            Desired number of shifted nodes. Actual number can be lower.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two datasets with shape [n, d]
            Dataset X has normal noise N(0,1), while dataset Y has shifts based on the specified case.

        """
        self.n = n
        self.adj_env1 = self._simulate_dag(self.d, self.s0, self.graph_type)
        self.shifted_nodes = self._choose_shifted_nodes(self.adj_env1, num_shifted_nodes)
        self.adj_env2 = self.adj_env1.copy()
        data_env1, data_env2 = self._sample_from_same_structs(self.adj_env1, self.shifted_nodes)
        
        return data_env1, data_env2

    def plot_dag_with_shifts(self, filename: str = "dag_plot.png") -> None:
        """
        Save DAG figure to file, highlighting shifted nodes.
        
        Args:
            filename: Output filename for the plot
        """
        if not hasattr(self, 'adj_env1') or self.adj_env1 is None:
            raise RuntimeError("Call sample() first.")
            
        g = ig.Graph.Adjacency(self.adj_env1.tolist(), mode="directed")
        layout = g.layout("kk")

        colors = []
        labels = []
        for i in range(self.d):
            if i in self.sin_cos_shift_nodes:
                c = "red"  # sin->cos function shift
            elif i in self.pa_delete_shift_nodes:
                c = "orange"  # parent deletion
            elif i in self.noise_n02_shift_nodes:
                c = "green"  # N(0,2) noise shift
            elif i in self.noise_laplace_shift_nodes:
                c = "purple"  # Laplace noise shift
            else:
                c = "lightblue"  # no shift
                    
            lbl = str(i)
            colors.append(c)
            labels.append(lbl)

        g.vs["color"] = colors
        g.vs["label"] = labels
        g.vs["size"] = 30

        ig.plot(g,
               target=filename,
               layout=layout,
               bbox=(600, 400),
               margin=50,
               vertex_label_color="black")
        print(f"Saved DAG figure to '{filename}'")
