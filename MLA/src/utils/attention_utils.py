import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class AttentionHeadHook:
    def __init__(self, model):
        if hasattr(model, "_orig_mod"):
            self.model = model._orig_mod
        else:
            self.model = model
        
        self.handles = []
        self.activations = {}

        self.n_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.hidden_size // self.n_heads

    def hook_fn(self, layer_idx):
        def hook(module, inputs, outputs):
            context_layer = outputs[0]
            batch_size, seq_len, _ = context_layer.shape

            heads = (
                context_layer
                .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
                .permute(2, 0, 1, 3)
                .reshape(self.n_heads, -1, self.head_dim)
                .detach()
                .cpu()
            )
            self.activations[layer_idx] = heads

        return hook

    def register(self):
        for i, layer in enumerate(self.model.bert.encoder.layer):
            handle = layer.attention.self.register_forward_hook(
                self.hook_fn(i)
            )
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


def linear_cka(X, Y):
    """
    X, Y: (N, D) - Features for two heads (e.g., 46208 x 64)
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

def compute_head_similarity_matrix(head_data):
    """
    head_data: torch.Size([16, 46208, 64])
    """
    num_heads = head_data.shape[0]
    cka_matrix = np.zeros((num_heads, num_heads))
    
    for i in range(num_heads):
        for j in range(i, num_heads):
            score = linear_cka(head_data[i], head_data[j])
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score # Symmetric
            
    return cka_matrix

def plot_heatmap(ax, matrix, layer_id=None):
    """
    Plots the Linear CKA similarity matrix and calculates/displays the average head overlap.
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

def plot_all_layers_grid(matrix_list, cols=2):
    """
    Plots all CKA matrices in a single grid.
    """
    avg_overlap_list = []
    num_layers = len(matrix_list)
    rows = math.ceil(num_layers / cols)
    
    # Adjust figsize based on grid size (width, height)
    _, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
    axes = axes.flatten()
    
    for layer_id, matrix in enumerate(matrix_list):
        ax = axes[layer_id]
        ax, avg_overlap = plot_heatmap(ax, matrix, layer_id=layer_id)
        avg_overlap_list.append(avg_overlap)
    
    plt.tight_layout()
    plt.show()

    return avg_overlap_list

# # Register the hooks
# torch._dynamo.reset()
# collector = AttentionHeadHook(pretrainer.model)
# collector.register()
