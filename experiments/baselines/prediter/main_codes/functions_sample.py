#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU version of PreDITEr functions using NumPy
"""

import numpy as np
import numpy.linalg as LA

def soft_thresholding(x, alpha):
    """CPU version of soft thresholding"""
    return np.maximum((np.abs(x) - alpha), 0) * np.sign(x)

def Delta_Theta_func(S1, S2, lambda_l1=0.1, rho=1.0, n_max_iter=500, stop_cond=1e-6, verbose=False, return_sym=True):
    """CPU version of Delta_Theta_func"""
    
    p = len(S1)
    # Initialize
    Delta = np.zeros([p, p])
    Phi = np.zeros([p, p])
    Lambda = np.zeros([p, p])
    
    # Eigenvalue computation
    eigen_max = LA.eigvals(S1)[0] * LA.eigvals(S2)[0]
    eigen_min = LA.eigvals(S1)[-1] * LA.eigvals(S2)[-1]
    
    if rho is None:
        if lambda_l1 <= float(eigen_min):
            rho = float(eigen_min)
        elif lambda_l1 <= float(eigen_max):
            rho = float(eigen_max)
        else:
            rho = lambda_l1
    
    # SVD
    [U1, D1, _] = LA.svd(S1)
    [U2, D2, _] = LA.svd(S2)
    B = 1 / (D1[:, np.newaxis] * D2[np.newaxis, :] + rho)
    
    obj_hist = np.zeros(n_max_iter)
    
    # ADMM iterations
    for it in range(n_max_iter):
        A = (S1 - S2) - Lambda + rho * Phi
        # Update Delta
        Delta = U1 @ (B * (U1.T @ A @ U2)) @ U2.T
        # Update Phi
        Phi = soft_thresholding(Delta + Lambda / rho, lambda_l1 / rho)
        # Update Lambda
        Lambda += rho * (Delta - Phi)
        
        # Objective computation (simplified)
        obj = np.sum(np.abs(Phi)) * lambda_l1
        obj_hist[it] = obj
        
        # Check stopping condition
        if it > 0 and np.abs(obj - obj_hist[it-1]) < stop_cond * (np.abs(obj) + 1):
            if verbose:
                sparsity = np.mean(Phi != 0)
                print(f'CPU: Converged in {it} iterations, sparsity: {float(sparsity):.3f}')
            break
    
    # Return symmetric
    if return_sym:
        Phi = (Phi + Phi.T) / 2
    
    return Phi, obj_hist[:it+1]

def IMAG_pasp_sample(S1, S2, max_subset_size=None, n_max_iter=100, stop_cond=1e-4, tol=1e-6, verbose=False, return_pasp=True, lambda_1=0.1, lambda_pasp=0.1, rho=1.0, th1=0.001, only_diag=False):
    """
    IMAG_pasp_sample function for CPU - returns node indices of detected interventions
    
    This function implements the IMAG algorithm for detecting intervention targets
    by comparing covariance matrices between two environments.
    
    Parameters:
    -----------
    S1, S2 : numpy.ndarray
        Covariance matrices from two environments
    max_subset_size : int, optional
        Maximum subset size for subset selection
    n_max_iter : int
        Maximum number of iterations
    stop_cond : float
        Stopping condition for convergence
    tol : float
        Tolerance for numerical operations
    verbose : bool
        Whether to print verbose output
    return_pasp : bool
        Whether to return PASP information
    lambda_1 : float
        L1 regularization parameter
    lambda_pasp : float
        PASP regularization parameter
    rho : float
        ADMM parameter
    th1 : float
        Threshold parameter for intervention detection
    only_diag : bool
        Whether to consider only diagonal elements
    
    Returns:
    --------
    K : list
        List of detected intervention node indices
    K_pasp : list, optional
        List of PASP information if return_pasp=True
    """
    
    p = S1.shape[0]
    
    if max_subset_size is None:
        max_subset_size = min(p, 20)  # Default subset size
    
    # Initialize
    detected_nodes = []
    pasp_info = []
    
    # Main algorithm - check each node for interventions
    for i in range(p):
        if verbose and i % 10 == 0:
            print(f"Processing node {i}/{p}")
        
        # Select subset of nodes
        if only_diag:
            # Only consider diagonal (self-intervention)
            subset = [i]
        else:
            # Consider all other nodes
            subset = list(range(p))
            if i in subset:
                subset.remove(i)
            subset = subset[:max_subset_size]
        
        if len(subset) == 0:
            continue
        
        # Extract submatrices
        S1_sub = S1[np.ix_(subset, subset)]
        S2_sub = S2[np.ix_(subset, subset)]
        
        try:
            # Run Delta_Theta_func on subset
            Delta_sub, _ = Delta_Theta_func(S1_sub, S2_sub, lambda_1, rho, n_max_iter, stop_cond, verbose=False)
            
            # Check if this node shows intervention effects
            intervention_detected = False
            node_pasp = []
            
            # More sophisticated intervention detection
            if only_diag:
                # For diagonal case, check if the node itself shows significant change
                if abs(Delta_sub[0, 0]) > th1:
                    intervention_detected = True
                    node_pasp.append(i)
            else:
                # For off-diagonal case, check if any node in the subset affects node i
                # Look for significant changes in the precision matrix
                max_change = 0
                for j in range(len(subset)):
                    for k in range(len(subset)):
                        change = abs(Delta_sub[j, k])
                        if change > max_change:
                            max_change = change
                
                # Only detect intervention if the change is significant
                if max_change > th1:
                    intervention_detected = True
                    # Find which nodes in subset are most affected
                    for j, node_j in enumerate(subset):
                        if abs(Delta_sub[j, j]) > th1 * 0.5:  # Lower threshold for individual nodes
                            node_pasp.append(node_j)
            
            if intervention_detected:
                detected_nodes.append(i)
                if return_pasp:
                    pasp_info.append(node_pasp)
                            
        except Exception as e:
            if verbose:
                print(f"Error processing node {i}: {e}")
            continue
    
    if return_pasp:
        return detected_nodes, pasp_info
    else:
        return detected_nodes

# Additional utility functions
def get_intervention_nodes(K, threshold=0.001):
    """Extract intervention nodes from K matrix"""
    intervention_nodes = []
    for i in range(K.shape[0]):
        if np.any(np.abs(K[i, :]) > threshold) or np.any(np.abs(K[:, i]) > threshold):
            intervention_nodes.append(i)
    return intervention_nodes

def compute_precision_recall(K_true, K_pred, threshold=0.001):
    """Compute precision and recall for intervention detection"""
    true_positives = np.sum((np.abs(K_true) > threshold) & (np.abs(K_pred) > threshold))
    false_positives = np.sum((np.abs(K_true) <= threshold) & (np.abs(K_pred) > threshold))
    false_negatives = np.sum((np.abs(K_true) > threshold) & (np.abs(K_pred) <= threshold))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
