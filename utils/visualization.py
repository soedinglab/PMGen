import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import numpy as np

alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

def softmax(x, axis=None):
    # Subtract the maximum value for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x) #- x_max
    # Compute softmax
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    return softmax_x


def plot_and_save_heatmap(input_data, output_path, npz_key=None, batch=None, cmap='viridis', figsize=(10, 8), dpi=300,
                          log_prob=False, use_alphabet=True):
    """
    Plots a heatmap from a numpy array or file and saves it to the specified output path.
    Args:
    input_data (str or np.ndarray): Path to the input numpy array (.npy or .npz) or numpy array
    output_path (str): Path to save the output heatmap image
    npz_key (str, optional): Key to use when loading from .npz file
    batch (int, optional): Batch index to use if the input is a batch of data
    cmap (str, optional): Colormap to use for the heatmap. Default is 'viridis'
    figsize (tuple, optional): Figure size (width, height) in inches. Default is (10, 8)
    dpi (int, optional): DPI for the output image. Default is 300
    log_prob (bool): if true, a softmax function is implemented before plotting.
    use_alphabet (bool): if true, use alphabet labels for x-axis. Default is True.
    """
    # Load the data if it's a file path, otherwise use the input directly
    if isinstance(input_data, str):
        if input_data.endswith('.npz'):
            if npz_key is None:
                raise ValueError("npz_key must be provided for .npz files")
            data = np.load(input_data)[npz_key]
        else:
            data = np.load(input_data)
    else:
        data = input_data
    if isinstance(batch, int):
        data = data[batch]
    elif batch == 'all':
        data = np.mean(data, axis=0, keepdims=False)
    # Apply softmax if log_prob is True
    if log_prob:
        print(data)
        data = np.exp(data)

    # Create the plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data, cmap=cmap, cbar_kws={'label': 'Value'})
    if use_alphabet:
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        num_cols = data.shape[1]
        if num_cols <= len(alphabet):
            x_labels = list(alphabet[:num_cols])
        else:
            x_labels = [alphabet[i % len(alphabet)] for i in range(num_cols)]
        ax.set_xticks(np.arange(len(x_labels)) + 0.5)
        ax.set_xticklabels(x_labels)
    plt.title('Heatmap')
    plt.tight_layout()
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the plot
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    print(f"Heatmap saved to {output_path}")

input_data = '/home/amir/amir/ParseFold/ParseFold-MHC/ProteinMPNN/outputs/default_example_no_fixed/probs/3HTN.npz'
output_path = '/home/amir/amir/ParseFold/ParseFold-MHC/ProteinMPNN/outputs/default_example_no_fixed/heatmap_probs.png'
plot_and_save_heatmap(input_data, output_path, npz_key='probs', batch='all', cmap='viridis', figsize=(10, 8), dpi=600, log_prob=True)

'''
import numpy as np

# Define parameters
L = 9   # Peptide length (5 positions)
K = 10  # Number of samples (10 different probability distributions)
num_aa = 21  # 21 amino acids
# Generate random probability distributions for each position
# Shape: (K, L, num_aa) -> 10 samples, 5 positions, 21 amino acids
prob_distributions = np.random.dirichlet(np.ones(num_aa), size=(K, L))
# Compute the mean distribution for each position across the 10 samples
mean_distribution = np.mean(prob_distributions, axis=0)  # Shape: (L, num_aa)
# Compute entropy for each position (averaged over 10 samples) KL(P||Q) = sum(p_i * log(p_i/q_i))
entropy_per_position = np.mean(-np.sum(prob_distributions * np.log(prob_distributions + 1e-9), axis=2), axis=0)
# Compute KL divergence for each position (average KL divergence from each sample to the mean)
kl_divergences = np.sum(prob_distributions * (np.log(prob_distributions + 1e-9) - np.log(mean_distribution + 1e-9)), axis=2)
kl_per_position = np.mean(kl_divergences, axis=0)
# Print results
print("Entropy at each position:", entropy_per_position)
print("KL Divergence at each position:", kl_per_position)'''


