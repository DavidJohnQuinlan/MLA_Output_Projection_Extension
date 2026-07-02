import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Linear Centered Kernel Alignment (CKA) similarity between two feature matrices.

    CKA measures representational similarity between two sets of activations, invariant
    to orthogonal transformations and isotropic scaling. A score of 1.0 indicates identical
    representations; 0.0 indicates orthogonal representations.

    Args:
        X (torch.Tensor): Feature matrix of shape `(N, D)` for the first attention head.
        Y (torch.Tensor): Feature matrix of shape `(N, D)` for the second attention head.

    Returns:
        torch.Tensor: Scalar CKA similarity score in the range `[0, 1]`.
    """

    # Center the columns (features)
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    
    # Compute HSIC: Frobenius norm squared of (Y^T @ X)
    # This results in a small (D x D) matrix multiplication
    dot_product = torch.linalg.norm(Y.T @ X, ord='fro')**2
    
    # Compute Normalization (Denominators)
    norm_x = torch.linalg.norm(X.T @ X, ord='fro')
    norm_y = torch.linalg.norm(Y.T @ Y, ord='fro')
    
    return dot_product / (norm_x * norm_y)


def compute_head_similarity_matrix(head_data: torch.Tensor) -> np.ndarray:
    """
    Computes a pairwise Linear CKA similarity matrix across all attention heads in a layer.

    Args:
        head_data (torch.Tensor): Stacked head activations of shape `(n_heads, N, head_dim)`.

    Returns:
        np.ndarray: Symmetric similarity matrix of shape `(n_heads, n_heads)` with values in `[0, 1]`.
    """
    num_heads = head_data.shape[0]
    cka_matrix = np.zeros((num_heads, num_heads))
    
    for i in range(num_heads):
        for j in range(i, num_heads):
            score = linear_cka(head_data[i], head_data[j])
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score
            
    return cka_matrix


def compute_model_cka(activations_list: list[torch.Tensor]) -> float:
    """
    Computes the mean pairwise CKA head overlap across all encoder layers.

    For each layer, builds a pairwise CKA similarity matrix across attention heads
    and extracts the upper-triangle values to compute a per-layer average overlap.
    Returns the mean of these per-layer scores as a single model-level summary statistic.

    Args:
        activations_list (list[torch.Tensor]): Per-layer head activations, where each
            element has shape `(n_heads, N, head_dim)`.

    Returns:
        float: Mean CKA head overlap score across all layers, in the range `[0, 1]`.
    """    
    avg_layer_overlap = []
    for layer in activations_list:

        # Calculate layers head similarity
        layer_head_similarity = compute_head_similarity_matrix(layer)

        # Calculate overlap
        num_heads = layer_head_similarity.shape[0]
        upper_tri = layer_head_similarity[np.triu_indices(num_heads, k=1)]
        avg_layer_overlap.append(upper_tri.mean())

    return float(np.mean(avg_layer_overlap))


def plot_heatmap(ax: plt.Axes, matrix: np.ndarray, layer_id: int | None = None) -> tuple[plt.Axes, float]:
    """
    Plots a CKA similarity matrix as a heatmap on a given axes object.

    Args:
        ax (plt.Axes): Matplotlib axes to plot on.
        matrix (np.ndarray): Square similarity matrix of shape `(n_heads, n_heads)`.
        layer_id (int, optional): Layer index used in the subplot title. Defaults to None.

    Returns:
        tuple[plt.Axes, float]: The updated axes and the mean upper-triangle overlap score.
    """
    # Calculate overlap
    num_heads = matrix.shape[0]
    upper_tri = matrix[np.triu_indices(num_heads, k=1)]
    avg_overlap = float(f"{upper_tri.mean():.4f}")
    
    # Plot heatmap on the specific subplot axis
    sns.heatmap(matrix, annot=True, cmap='viridis', fmt=".2f", vmin=0, vmax=1, ax=ax, cbar=True)
    ax.set_title(f"Layer {layer_id}\nAvg Overlap: {avg_overlap:.4f}", fontsize=12)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Head Index")

    return ax, avg_overlap


def plot_all_layers_grid(activations_list, cols: int = 2) -> list[float]:
    """
    Plots CKA similarity matrices for all encoder layers in a single grid figure.

     Args:
        activations_list (list[np.ndarray]): One similarity matrix per encoder layer.
        cols (int): Number of columns in the grid layout. Defaults to 2.

    Returns:
        list[float]: Average head overlap score per layer.
    """
    num_layers = len(activations_list)
    rows = math.ceil(num_layers / cols)
    
    # Adjust figsize based on grid size (width, height)
    _, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
    axes = axes.flatten()
    avg_overlap_list = []
    for layer_id, matrix in enumerate(activations_list):
        similarty_matrix = compute_head_similarity_matrix(matrix)
        ax = axes[layer_id]
        ax, avg_overlap = plot_heatmap(ax, similarty_matrix, layer_id=layer_id)
        avg_overlap_list.append(avg_overlap)
    
    plt.tight_layout()
    plt.show()

    return avg_overlap_list
