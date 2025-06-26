#!/usr/bin/env python
"""
=========================

End‑to‑end trainer for a **peptide×MHC SCQ-VAE clustering**.
For each fold, load cross_latents.npz containing cross-atnn and mhc_ids, and labels.
train and evaluate SCQ-VAE model on the cross-attn data.
Visualize the results with t-SNE and UMAP with mhc_ids and labels.

Author: Amirreza (memory-optimized version, 2025)
"""
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm
from utils.model import SCQ1DAutoEncoder
from sklearn.cluster import KMeans

# Enable eager execution explicitly to resolve graph mode issues
tf.config.run_functions_eagerly(True)
from tensorflow import keras

# Global random state for reproducibility
random_state = 42

def load_cross_latents_data(file_path):
    """Load cross-attention data from a .npz file."""
    data = np.load(file_path)
    cross_latents = data['cross_latents']
    mhc_ids = data['mhc_ids']
    labels = data['labels']
    return cross_latents, mhc_ids, labels # (N, seq_length, embedding_dim), (N,), (N,)

def create_dataset(cross_latents, labels=None):
    """Create a TensorFlow dataset from cross-latents data."""
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(tf.cast(cross_latents, tf.float32))
    else:
        # Cast features to float32
        features = tf.cast(cross_latents, tf.float32)
        labels = tf.cast(labels, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    return dataset

def initialize_codebook_with_kmeans(X_train, num_embeddings, embedding_dim):
    """Initialize codebook vectors using k-means clustering."""
    kmeans = KMeans(n_clusters=num_embeddings, random_state=random_state)
    flat_data = X_train.reshape(-1, X_train.shape[-1])
    kmeans.fit(flat_data)
    return kmeans.cluster_centers_.astype(np.float32)

# Visualization functions
def plot_training_metrics(history, save_path=None):
    """Plot training and validation loss and accuracy."""
    return


def plot_reconstructions(original, reconstructed, n_samples=5, save_path=None, visualize=True):
    """Plot comparison between original and reconstructed sequences."""
    n_samples = min(n_samples, len(original))
    plt.figure(figsize=(15, 3 * n_samples))
    for i in range(n_samples):
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.plot(original[i])
        plt.title(f"Original Sequence {i + 1}")
        plt.grid(True)
        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.plot(reconstructed[i])
        plt.title(f"Reconstructed Sequence {i + 1}")
        plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if visualize:
        plt.show()
    else:
        plt.close()


def plot_codebook_usage(indices, num_embeddings, save_path=None, visualize=True):
    """Visualize the usage distribution of codebook vectors.

    Args:
        indices: Hard indices from model output (integer indices of assigned codes)
        num_embeddings: Total number of vectors in the codebook
        save_path: Path to save the visualization

    Returns:
        used_vectors: Number of vectors used
        usage_percentage: Percentage of codebook utilized
    """
    # Convert from tensor to numpy if needed
    if isinstance(indices, tf.Tensor):
        indices = indices.numpy()

    # Ensure indices is a NumPy array before proceeding
    if not isinstance(indices, np.ndarray):
        try:
            # Attempt conversion if it's list-like
            indices = np.array(indices)
        except Exception as e:
            print(
                f"Error in plot_codebook_usage: Input 'indices' is not a NumPy array or convertible. Type: {type(indices)}. Error: {e}")
            # Return default values or raise an error
            return 0, 0.0

    # Flatten indices to 1D array for counting
    try:
        flat_indices = indices.flatten()
    except AttributeError:
        print(f"Error in plot_codebook_usage: Cannot flatten 'indices'. Type: {type(indices)}")
        return 0, 0.0  # Return default values

    # Count occurrences of each codebook vector
    try:
        unique, counts = np.unique(flat_indices, return_counts=True)
    except TypeError as e:
        print(
            f"Error in plot_codebook_usage: Cannot compute unique values for 'flat_indices'. Type: {type(flat_indices)}. Error: {e}")
        return 0, 0.0  # Return default values

    # Create full distribution including zeros for unused vectors
    full_distribution = np.zeros(num_embeddings)
    for idx, count in zip(unique, counts):
        if 0 <= idx < num_embeddings:  # Ensure index is valid
            full_distribution[int(idx)] = count

    # Calculate usage statistics
    used_vectors = np.sum(full_distribution > 0)
    usage_percentage = (used_vectors / num_embeddings) * 100

    # Create the plot
    plt.figure(figsize=(12, 6))
    bar_positions = np.arange(num_embeddings)
    bars = plt.bar(bar_positions, full_distribution)

    # Color bars by frequency
    max_count = np.max(full_distribution)
    if max_count > 0:
        for i, bar in enumerate(bars):
            intensity = full_distribution[i] / max_count
            bar.set_color(plt.cm.viridis(intensity))

    plt.xlabel('Codebook Vector Index')
    plt.ylabel('Usage Count')
    plt.title(
        f'Codebook Vector Usage Distribution\n{used_vectors}/{num_embeddings} vectors used ({usage_percentage:.1f}%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Frequency')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    print(f"Codebook usage: {used_vectors}/{num_embeddings} vectors used ({usage_percentage:.1f}%)")
    if visualize:
        plt.show()
    else:
        plt.close()
    return used_vectors, usage_percentage


def plot_soft_cluster_distribution(soft_cluster_probs, num_samples=None, save_path=None, visualize=True):
    """
    Visualize the distribution of soft clustering probabilities across samples.

    Args:
        soft_cluster_probs: List of arrays or single array containing probability
                           distributions across clusters for each sample
        num_samples: Number of samples to visualize (default: 10)
        save_path: Path to save the visualization (default: None)

    Returns:
        None
    """
    if num_samples is None:
        num_samples = len(soft_cluster_probs)
    # Process input to get a clean 2D array (samples x clusters)
    if isinstance(soft_cluster_probs, list):
        if not soft_cluster_probs:
            print("Warning: Empty list provided to plot_soft_cluster_distribution")
            return
        sample_batch = soft_cluster_probs[0]
        if isinstance(sample_batch, tf.Tensor):
            sample_batch = sample_batch.numpy()
        soft_cluster_probs = sample_batch
    elif isinstance(soft_cluster_probs, tf.Tensor):
        soft_cluster_probs = soft_cluster_probs.numpy()

    # Reshape if needed
    if soft_cluster_probs.ndim > 2:
        print(f"Reshaping array from shape {soft_cluster_probs.shape} to 2D format")
        if soft_cluster_probs.shape[1] == 1:
            soft_cluster_probs = soft_cluster_probs.reshape(soft_cluster_probs.shape[0],
                                                            soft_cluster_probs.shape[2])
        else:
            soft_cluster_probs = soft_cluster_probs.reshape(soft_cluster_probs.shape[0], -1)

    # Verify valid data
    if soft_cluster_probs.size == 0:
        print("Warning: Empty array provided to plot_soft_cluster_distribution")
        return

    n_samples = min(len(soft_cluster_probs), 1000)  # Limit to prevent memory issues
    n_clusters = soft_cluster_probs.shape[1]

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

    # 1. Heatmap visualization (top)
    display_samples = min(num_samples, n_samples)
    im = ax1.imshow(soft_cluster_probs[:display_samples], aspect='auto', cmap='viridis')
    ax1.set_xlabel('Cluster Index')
    ax1.set_ylabel('Sample Index')
    ax1.set_title(f'Soft Cluster Probability Heatmap (First {display_samples} Samples)')
    plt.colorbar(im, ax=ax1, label='Probability')

    # 2. Aggregated statistics (bottom)
    # Calculate statistics across all samples for each cluster
    cluster_means = np.mean(soft_cluster_probs, axis=0)
    cluster_max_counts = np.sum(np.argmax(soft_cluster_probs, axis=1)[:, np.newaxis] == np.arange(n_clusters), axis=0)

    # Create a twin axis for the bar plot
    ax2_twin = ax2.twinx()

    # Plot mean probability for each cluster (line)
    x = np.arange(n_clusters)
    ax2.plot(x, cluster_means, 'r-', linewidth=2, label='Mean Probability')
    ax2.set_ylabel('Mean Probability', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, max(cluster_means) * 1.2)

    # Plot histogram of cluster assignments (bars)
    ax2_twin.bar(x, cluster_max_counts, alpha=0.3, label='Assignment Count')
    ax2_twin.set_ylabel('Number of Samples\nwith Highest Probability', color='b')
    ax2_twin.tick_params(axis='y', labelcolor='b')

    # Add labels and grid
    ax2.set_xlabel('Cluster Index')
    ax2.set_title('Cluster Usage Statistics Across All Samples')
    ax2.set_xticks(np.arange(0, n_clusters, max(1, n_clusters // 20)))
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')

    # Create custom legend
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # Add overall statistics as text
    active_clusters = np.sum(np.max(soft_cluster_probs, axis=0) > 0.01)
    most_used_cluster = np.argmax(cluster_max_counts)
    ax2.text(0.02, 0.95,
             f"Active clusters: {active_clusters}/{n_clusters} ({active_clusters / n_clusters:.1%})\n"
             f"Most used cluster: {most_used_cluster} ({cluster_max_counts[most_used_cluster]} samples)",
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show or close based on global visualize flag
    if visualize:
        plt.show()
    else:
        plt.close()


def plot_cluster_distribution(soft_cluster_probs, save_path=None, visualize=True):
    """
    Plot distribution of samples across clusters based on one-hot encodings.

    Args:
        soft_cluster_probs: Soft cluster probabilities
        save_path: Path to save the plot

    Returns:
        used_clusters: Number of clusters used
        usage_percentage: Percentage of clusters used
    """
    # Convert soft cluster probabilities to one-hot encodings
    one_hot_encodings = tf.one_hot(tf.argmax(soft_cluster_probs, axis=-1), depth=soft_cluster_probs.shape[-1])
    one_hot_encodings = tf.cast(one_hot_encodings, tf.float32)

    # print(f"one_hot_encodings shape: {one_hot_encodings.shape}")
    # print first 5 values
    # print(f"one_hot_encodings values: {one_hot_encodings[:5]}")

    # Convert one-hot to cluster indices if needed
    if isinstance(one_hot_encodings, tf.Tensor):
        one_hot_encodings = one_hot_encodings.numpy()

    # Handle different shapes of one_hot_encodings
    if len(one_hot_encodings.shape) == 3:  # (batch, seq_len, num_embeddings)
        cluster_assignments = np.argmax(one_hot_encodings, axis=-1).flatten()
    else:  # (batch, num_embeddings)
        cluster_assignments = np.argmax(one_hot_encodings, axis=-1).flatten()

    # Count occurrences of each cluster
    unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)

    # Create a full distribution including zeros for unused clusters
    num_clusters = one_hot_encodings.shape[-1]
    full_distribution = np.zeros(num_clusters)
    for cluster, count in zip(unique_clusters, counts):
        full_distribution[cluster] = count

    # Calculate usage statistics
    used_clusters = np.sum(full_distribution > 0)
    usage_percentage = (used_clusters / num_clusters) * 100

    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(num_clusters), full_distribution)

    # Color bars by frequency
    max_count = np.max(full_distribution)
    if max_count > 0:
        for i, bar in enumerate(bars):
            intensity = full_distribution[i] / max_count
            bar.set_color(plt.cm.plasma(intensity))

    plt.xlabel('Cluster Index')
    plt.ylabel('Number of Samples')
    plt.title(
        f'Sample Distribution Across Clusters\n{used_clusters}/{num_clusters} clusters used ({usage_percentage:.1f}%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Sample Count')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    print(f"Cluster usage: {used_clusters}/{num_clusters} clusters contain samples ({usage_percentage:.1f}%)")
    if visualize:
        plt.show()
    else:
        plt.close()
    return used_clusters, usage_percentage

def plot_tsne_umap(cross_latents, mhc_ids, labels, save_path=None):
    """Plot t-SNE and UMAP visualizations of the cross-latents."""
    # Standardize the data
    scaler = StandardScaler()
    cross_latents_scaled = scaler.fit_transform(cross_latents)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_results = tsne.fit_transform(cross_latents_scaled)

    # Plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # t-SNE plot
    plt.subplot(1, 2, 1)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=5)
    plt.title('t-SNE Visualization')
    plt.colorbar(label='Labels')

    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")

    plt.show()


def plot_PCA(cross_latents, mhc_ids=None, labels=None, save_path=None):
    """Plot PCA visualization of the cross-latents with optional label highlighting."""

    # reduce dimensions if necessary
    if cross_latents.ndim > 2:
        # Flatten the cross_latents to ensure they are 2D (N, seq_length * embedding_dim)
        cross_latents = cross_latents.reshape(cross_latents.shape[0], -1)

    # Standardize the data
    scaler = StandardScaler()
    cross_latents_scaled = scaler.fit_transform(cross_latents)

    # PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(cross_latents_scaled)

    # Plotting
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 6))

    if mhc_ids is not None:
        # Convert string MHC IDs to categorical indices
        unique_mhcs = np.unique(mhc_ids)
        mhc_to_index = {mhc: i for i, mhc in enumerate(unique_mhcs)}
        numeric_mhc_ids = np.array([mhc_to_index[mhc] for mhc in mhc_ids])

        # Plot using the numeric encoding
        sc = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=numeric_mhc_ids, cmap='viridis', s=5)
        plt.colorbar(sc, label='MHC IDs')

        # If not too many unique MHCs, add a legend
        if len(unique_mhcs) <= 20:
            from matplotlib.lines import Line2D
            cmap = plt.cm.get_cmap('viridis', len(unique_mhcs))
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i),
                       label=str(mhc.decode('utf-8') if isinstance(mhc, bytes) else mhc), markersize=5)
                for i, mhc in enumerate(unique_mhcs)
            ]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Default plotting without colors
        plt.scatter(pca_results[:, 0], pca_results[:, 1], s=5)

    # Highlight positive labels (value 1)
    if labels is not None:
        # Flatten and convert labels if needed
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            flat_labels = np.asarray(labels).flatten()
            # Find indices of points with positive labels (value 1 or close to 1)
            positive_indices = np.where(np.isclose(flat_labels, 1.0, atol=0.01))[0]

            if len(positive_indices) > 0:
                # Plot circles around positive points
                plt.scatter(pca_results[positive_indices, 0], pca_results[positive_indices, 1],
                            s=30, facecolors='none', edgecolors='orange', linewidths=0.07,
                            label='Positive Labels')

                # Add a legend entry for positive labels
                plt.legend(loc='upper right')
                plt.title('PCA Visualization (Red circles: Positive Labels)')
            else:
                plt.title('PCA Visualization (No positive labels found)')
        else:
            plt.title('PCA Visualization')
    else:
        plt.title('PCA Visualization')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"PCA plot saved to {save_path}")

    plt.show()


def process_and_save(dataset: tf.data.Dataset,
                     split_name: str,
                     model: tf.keras.Model,
                     output_dir: str,
                     num_embeddings: int,
                     mhc_ids: np.ndarray = None,
                     labels: np.ndarray = None):
    """
    Quantize `dataset` through `model`, assemble into a DataFrame,
    save to parquet, and plot distributions with split-specific filenames.
    """
    quantized_latents = []
    cluster_indices_soft = []
    cluster_indices_hard = []
    labels = []

    # Extract latent codes for every batch
    for batch_X, batch_y in dataset:
        Zq, out_P_proj, _, _ = model.encode_(batch_X)
        quantized_latents.append(Zq.numpy())
        cluster_indices_soft.append(out_P_proj.numpy())
        cluster_indices_hard.append(tf.argmax(out_P_proj, axis=-1).numpy())
        labels.append(batch_y.numpy())

    # Concatenate across batches
    quantized_latent = np.concatenate(quantized_latents, axis=0)
    soft_probs = np.concatenate(cluster_indices_soft, axis=0)
    hard_assign = np.concatenate(cluster_indices_hard, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Build DataFrame
    records = []
    for i in range(len(quantized_latent)):
        rec = {}
        flat_latent = quantized_latent[i].flatten()
        for j, v in enumerate(flat_latent):
            rec[f'latent_{j}'] = float(v)

        flat_soft = soft_probs[i].flatten()
        for j, v in enumerate(flat_soft):
            rec[f'soft_cluster_{j}'] = float(v)

        rec['hard_cluster'] = int(hard_assign[i])
        # binding label if available
        if i < len(labels):
            lbl = labels[i]
            # Handle scalar, 0-dim, or 1-dim arrays
            if isinstance(lbl, (np.ndarray, list)) and np.asarray(lbl).size > 0:
                rec['binding_label'] = float(np.asarray(lbl).flatten()[0])
            else:
                rec['binding_label'] = float(lbl)
        else:
            rec['binding_label'] = np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save parquet
    parquet_path = os.path.join(output_dir, f'quantized_outputs_{split_name}.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"[{split_name}] saved parquet → {parquet_path}")

    # Plot distributions
    plot_cluster_distribution(soft_probs,
                              save_path=os.path.join(output_dir, f'cluster_distribution_soft_{split_name}.png'))
    plot_codebook_usage(hard_assign, num_embeddings,
                        save_path=os.path.join(output_dir, f'codebook_usage_{split_name}.png'))
    plot_soft_cluster_distribution(soft_probs, 20,
                                   save_path=os.path.join(output_dir,
                                                          f'soft_cluster_distribution_{split_name}.png'))

    # visualize PCA with label highlights
    print(quantized_latent.shape)
    # Pass both mhc_ids and labels to plot_PCA
    plot_PCA(quantized_latent,
             save_path=os.path.join(output_dir, f'quantized_latents_pca_{split_name}.png'),
             mhc_ids=mhc_ids,
             labels=labels)  # Add labels parameter



    # visualize by hard cluster assignments
    plot_PCA(quantized_latent,
             save_path=os.path.join(output_dir, f'quantized_latents_pca_hard_{split_name}.png'),
             mhc_ids=hard_assign.flatten(),
             labels=labels)  # Add labels parameter



def main():
    # --- Configuration ---
    num_embeddings = 16  # Number of embeddings in the codebook
    embedding_dim = 4  # Dimension of each embedding vector
    commitment_beta = 0.25  # Commitment loss weight
    batch_size = 32  # Batch size for training
    epochs = 3  # Number of training epochs
    learning_rate = 1e-3  # Learning rate for the optimizer

    cross_latents_file = 'runs/run_20250626-114618/cross_latent_test1_fold_1.npz'  # Path to the cross-attention data file
    val_file = 'runs/run_20250626-114618/cross_latent_test2_fold_1.npz'  # Path to the validation data file (if needed)
    save_dir = 'runs/run_20250626-114618/scq'  # Directory to save results and plots
    os.makedirs(save_dir, exist_ok=True)

    # --- Load Cross-Attention Data ---
    print("Loading cross-attention data...")
    cross_latents, mhc_ids, labels = load_cross_latents_data(cross_latents_file)
    print("Cross-latents shape:", cross_latents.shape)
    # flatten the cross_latents to ensure they are 2D (N, seq_length * embedding_dim)
    cross_latents = cross_latents.reshape(cross_latents.shape[0], -1)  # Flatten to (N, seq_length * embedding_dim)
    seq_length = cross_latents.shape[1]  # Length of the sequences

    val_latents, val_mhc_ids, val_labels = load_cross_latents_data(val_file)
    if val_latents is not None:
        print("Validation cross-latents shape:", val_latents.shape)
        val_latents = val_latents.reshape(val_latents.shape[0], -1)
    else:
        print("No validation data found, proceeding without validation set.")
        val_latents, val_mhc_ids, val_labels = None, None, None


    # --- Create TensorFlow Dataset ---
    print("Creating TensorFlow dataset...")
    dataset = create_dataset(cross_latents, labels)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # drop labels
    dataset_train = create_dataset(cross_latents)
    dataset_train = dataset_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Batch and prefetch for performance
    print("Dataset created with {} batches.".format(len(dataset)))
    # --- Print Dataset Information ---
    print("Cross-latents shape:", cross_latents.shape)
    print("MHC IDs shape:", mhc_ids.shape)
    print("Labels shape:", labels.shape)
    print("Sequence length:", seq_length)

    # Visualize raw cross-latents data
    print("Visualizing raw cross-latents data with PCA...")
    plot_PCA(cross_latents, mhc_ids, save_path=os.path.join(save_dir, 'cross_latents_pca.png'), labels=labels)
    print("Visualizing cross-latents data with t-SNE and UMAP...")

    # --- Initialize Codebook ---
    print("Initializing codebook with k-means...")
    codebook_init = initialize_codebook_with_kmeans(cross_latents, num_embeddings, embedding_dim)
    print("Codebook initialized with shape:", codebook_init.shape)

    # --- Model Instantiation ---
    print("Building the SCQ1DAutoEncoder model...")
    input_shape = (seq_length,)
    # Print detailed information about input dimensions
    print(f"Input shape being passed to model: {input_shape}")
    print(f"Expected embedding_dim: {embedding_dim}")
    print(f"Actual shape of cross_latents: {cross_latents.shape}")

    # Check if the embedding_dim needs adjustment based on the actual data
    if embedding_dim != cross_latents.shape[-1]:
        print(f"Warning: Adjusting embedding_dim from {embedding_dim} to match data: {cross_latents.shape[-1]}")
        embedding_dim = cross_latents.shape[-1]

    model = SCQ1DAutoEncoder(
        input_dim=input_shape,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_beta=commitment_beta,
        scq_params={
            'lambda_reg': 1.0,
            'discrete_loss': False,
            'reset_dead_codes': True,
            'usage_threshold': 1e-4,
            'reset_interval': 5
        },
        cluster_lambda=1
    )
    print("Model built.")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # --- Training the Model ---
    print("Starting training...")
    print(f"Training on balanced set for {epochs} epochs...")
    start_time = time.time()
    history = model.fit(
        dataset_train,
        validation_data=val_latents,
        epochs=epochs,
    )
    end_time = time.time()
    print(f"Balanced training finished in {end_time - start_time:.2f} seconds.")

    # --- Save the Model ---
    model_save_path = os.path.join(save_dir, 'scq_vae_model.h5')
    print(f"Saving the model to {model_save_path}...")
    model.save(model_save_path)
    print("Model saved successfully.")

    # --- Evaluate the Model ---
    print("Evaluating the model...")
    val_latents = None
    val_dataset = None
    if val_latents is not None:
        val_dataset = create_dataset(val_latents, val_labels).batch(batch_size)
        evaluation_results = model.evaluate(val_dataset)
        print(f"Validation Loss: {evaluation_results[0]}, Validation Accuracy: {evaluation_results[1]}")
    else:
        print("No validation data provided, skipping evaluation.")
    print("Model evaluation completed.")

    # --- Visualize Results ---
    print("Visualizing results...")
    process_and_save(dataset, 'train', model, save_dir, num_embeddings, mhc_ids)
    if val_dataset is not None:
        process_and_save(val_dataset, 'val', model, save_dir, num_embeddings, val_mhc_ids)

    print("Results visualization is not implemented yet.")
    print("End-to-end training completed successfully.")
    print("You can now proceed with further analysis or visualization of the model outputs.")
    # --- End of Main Function ---

if __name__ == "__main__":
    main()