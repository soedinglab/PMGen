# load npz files and print the shape of the data
import numpy as np
import pandas as pd
import os

def load_npz_files(directory):
    keys = []
    embeddings = []
    sequences = []
    # load sequence from the csv file
    df = pd.read_csv(os.path.join(directory, 'sequences.csv'))
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            print("Loading file:", filename)
            file_path = os.path.join(directory, filename)
            with np.load(file_path) as npz_file:
                print(f"{filename}: contains {len(npz_file.files)} arrays")
                for key in npz_file.files:
                    embedding = npz_file[key]
                    if embedding.shape[0] == 187 and embedding.shape[1] == 1152:
                        embeddings.append(embedding)
                        sequences.append(df[df['id'] == key]['sequence'].values[0])
                    else:
                        print(f"Skipping {key} due to unexpected shape: {embedding.shape}")

                    keys.append(key)

    return embeddings, keys, sequences

def print_shapes(data, keys, seqs):
    for arr, key, seq in zip(data, keys, seqs):
        print(f"Shape of array {key}: {arr.shape} with sequence {seq}")

def print_samples(data, keys):
    for arr, key in zip(data, keys):
        print(f"Sample of array {key}: {arr[:1]}")  # Print first samples

def plot_pairwise_comparison(selected_keys, keys, data, seqs):
    """
    Compare sequence and embedding similarities between selected pairs of protein sequences.

    Args:
        selected_keys: List of key pairs to compare (e.g., [(key1, key2), (key3, key4)])
        keys: All available keys
        data: Embedding data (36, 1152)
        seqs: Protein sequences
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cosine
    import seaborn as sns
    from Levenshtein import distance as levenshtein_distance

    for key1, key2 in selected_keys:
        # Get indices of the keys in the main arrays
        idx1 = keys.index(key1)
        idx2 = keys.index(key2)

        # Get sequences and embeddings
        seq1, seq2 = seqs[idx1], seqs[idx2]
        emb1, emb2 = data[idx1], data[idx2]

        # Calculate sequence similarity
        seq_distance = levenshtein_distance(seq1, seq2)
        seq_similarity = 1 - (seq_distance / max(len(seq1), len(seq2)))

        # Calculate embedding similarity metrics
        cosine_sim = 1 - cosine(emb1.flatten(), emb2.flatten())
        euclidean_dist = np.linalg.norm(emb1 - emb2)

        # Calculate positional differences
        position_diff = np.abs(emb1 - emb2)  # Shape: (36, 1152)
        mean_diff_by_pos = position_diff.mean(axis=1)  # Average diff across 1152 dims for each position
        max_diff_by_pos = position_diff.max(axis=1)    # Max diff across 1152 dims for each position

        # Calculate feature differences
        feature_diff = position_diff.mean(axis=0)      # Average diff across 36 positions for each feature
        top_features_idx = np.argsort(feature_diff)[::-1][:20]  # Top 20 most different features

        # Create visualization
        fig, axs = plt.subplots(2, 2, figsize=(18, 14))
        plt.suptitle(f'Comparison: {key1} vs {key2}', fontsize=16)

        # 1. Heatmap of differences across all positions and features
        im = axs[0, 0].imshow(position_diff, aspect='auto', cmap='viridis')
        axs[0, 0].set_title(f'Embedding Differences (Sequence Sim: {seq_similarity:.2f}, Cosine Sim: {cosine_sim:.2f})')
        axs[0, 0].set_xlabel('Feature Dimension')
        axs[0, 0].set_ylabel('Sequence Position')
        plt.colorbar(im, ax=axs[0, 0], label='Absolute Difference')

        # 2. Bar plot of position differences (which positions differ most)
        axs[0, 1].bar(range(len(mean_diff_by_pos)), mean_diff_by_pos, alpha=0.7, label='Mean Diff')
        axs[0, 1].bar(range(len(max_diff_by_pos)), max_diff_by_pos, alpha=0.5, label='Max Diff')
        axs[0, 1].set_title('Differences by Position')
        axs[0, 1].set_xlabel('Position in Sequence')
        axs[0, 1].set_ylabel('Difference Magnitude')
        axs[0, 1].legend()

        # 3. Top different features
        axs[1, 0].bar(range(len(top_features_idx)), feature_diff[top_features_idx])
        axs[1, 0].set_title('Top 20 Most Different Features')
        axs[1, 0].set_xlabel('Feature Rank')
        axs[1, 0].set_ylabel('Mean Difference')
        axs[1, 0].set_xticks(range(len(top_features_idx)))
        axs[1, 0].set_xticklabels([f"{i}" for i in top_features_idx], rotation=90)

        # # 4. Correlation matrix between the two embeddings
        # corr_matrix = np.corrcoef(emb1, emb2)
        # sns.heatmap(corr_matrix, ax=axs[1, 1], cmap='coolwarm', vmin=-1, vmax=1)
        # axs[1, 1].set_title('Correlation Between Embeddings')
        #
        # plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.savefig(f'comparison_{key1}_vs_{key2}.png', dpi=300)
        # print(f"Saved comparison to comparison_{key1}_vs_{key2}.png")
        #
        # # Print summary
        # print(f"\nComparison of {key1} vs {key2}:")
        # print(f"  Sequence similarity: {seq_similarity:.4f}")
        # print(f"  Embedding cosine similarity: {cosine_sim:.4f}")
        # print(f"  Embedding euclidean distance: {euclidean_dist:.4f}")
        # print(f"  Top 5 most different positions: {np.argsort(mean_diff_by_pos)[::-1][:5]}")
        # print(f"  Top 5 most different features: {top_features_idx[:5]}")

        # 4. Sequence comparison visualization
        ax = axs[1, 1]
        ax.set_title('Sequence Comparison')
        ax.axis('off')  # Turn off axes

        # Find differences between sequences
        differences = []
        for i, (a, b) in enumerate(zip(seq1, seq2)):
            if a != b:
                differences.append((i, a, b))

        # Calculate sequence identity
        seq_len = max(len(seq1), len(seq2))
        percent_identity = (seq_len - len(differences)) / seq_len * 100

        # Show sequence statistics
        ax.text(0.05, 0.95, f"Sequence length: {len(seq1)} and {len(seq2)} amino acids", fontsize=11)
        ax.text(0.05, 0.9, f"Sequence identity: {percent_identity:.1f}%", fontsize=11)
        ax.text(0.05, 0.85, f"Number of differences: {len(differences)}", fontsize=11)

        # Display sequence alignment based on sequence length
        if len(seq1) > 80:  # For longer sequences, show regions with differences
            if differences:
                ax.text(0.05, 0.75, "Key differences:", fontsize=11, fontweight='bold')
                y_pos = 0.7

                # Show up to 5 difference regions
                for i, (pos, aa1, aa2) in enumerate(differences[:5]):
                    context_start = max(0, pos - 5)
                    context_end = min(len(seq1), pos + 6)

                    # Extract regions around differences
                    region1 = seq1[context_start:context_end]
                    region2 = seq2[context_start:context_end]

                    # Create marker pointing to difference
                    marker = " " * (pos - context_start) + "^"

                    ax.text(0.05, y_pos, f"Diff {i + 1} (pos {pos + 1}):", fontsize=10)
                    ax.text(0.05, y_pos - 0.05, region1, fontfamily='monospace', fontsize=10)
                    ax.text(0.05, y_pos - 0.1, region2, fontfamily='monospace', fontsize=10)
                    ax.text(0.05, y_pos - 0.15, marker, fontfamily='monospace', fontsize=10)

                    y_pos -= 0.2

                if len(differences) > 5:
                    ax.text(0.05, y_pos, f"...and {len(differences) - 5} more differences", fontsize=10)
            else:
                ax.text(0.05, 0.7, "The sequences are identical.", fontsize=11)
        else:
            # For shorter sequences, show complete alignment
            y_pos = 0.75
            ax.text(0.05, y_pos, "Sequence alignment:", fontsize=11, fontweight='bold')
            y_pos -= 0.05

            # Split into chunks of 50 for better display
            chunk_size = 50
            for i in range(0, len(seq1), chunk_size):
                chunk1 = seq1[i:i + chunk_size]
                chunk2 = seq2[i:i + chunk_size] if i < len(seq2) else ""

                # Create marker for differences
                marker = ""
                for j in range(len(chunk1)):
                    if j < len(chunk2) and chunk1[j] != chunk2[j]:
                        marker += "^"
                    else:
                        marker += " "

                ax.text(0.05, y_pos, f"Pos {i + 1}:", fontsize=9)
                y_pos -= 0.05
                ax.text(0.05, y_pos, chunk1, fontfamily='monospace', fontsize=10)
                y_pos -= 0.05
                ax.text(0.05, y_pos, chunk2, fontfamily='monospace', fontsize=10)
                y_pos -= 0.05
                ax.text(0.05, y_pos, marker, fontfamily='monospace', fontsize=10)
                y_pos -= 0.05

def plot_3d_PCA(data, keys, seqs=None, num_samples=100):
    """
    Plot 3D PCA visualization of the embeddings.

    Args:
        data: List of embeddings with shape (36, 1152)
        keys: List of identifiers for each embedding
        seqs: Optional list of sequences for annotation
        num_samples: Number of samples to plot (default: 10)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    import numpy as np
    import random

    # Limit to num_samples if there are more samples
    if len(data) > num_samples:
        # Select random indices
        indices = random.sample(range(len(data)), num_samples)
        sampled_data = [data[i] for i in indices]
        sampled_keys = [keys[i] for i in indices]
        sampled_seqs = [seqs[i] for i in indices] if seqs is not None else None
    else:
        sampled_data = data
        sampled_keys = keys
        sampled_seqs = seqs

    # Convert embeddings to feature vectors
    feature_vectors = []
    for embedding in sampled_data:
        # Mean across features to get (36,)
        feature_vectors.append(embedding.mean(axis=1))

    feature_vectors = np.array(feature_vectors)

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(feature_vectors)

    # Create 3D plot with larger size
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Use a colormap with distinct colors for each sample
    cmap = plt.cm.get_cmap('tab10', num_samples)

    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        pca_result[:, 2],
        c=range(len(pca_result)),  # Each sample gets its own color
        cmap=cmap,
        s=50,  # Smaller dot size
        alpha=0.8
    )

    # Add annotations for all points
    for i, key in enumerate(sampled_keys):
        ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], key, fontsize=9)

    # Set labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    ax.set_title(f'3D PCA of Protein Embeddings ({num_samples} Samples)', fontsize=16)

    # Add a color bar with sample IDs
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, ticks=range(len(sampled_keys)))
    cbar.set_label('Sample ID')
    cbar.ax.set_yticklabels(sampled_keys)

    # Add total explained variance as text
    total_var = sum(pca.explained_variance_ratio_[:3])
    fig.text(0.5, 0.01, f'Total variance explained: {total_var:.2%}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('3d_pca_embeddings_10samples.png', dpi=300)
    print(f"Saved 3D PCA visualization to 3d_pca_embeddings_10samples.png")

    # Create interactive plot if plotly is available
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'PC3': pca_result[:, 2],
            'Sample': sampled_keys
        })

        if sampled_seqs is not None:
            df['Sequence'] = sampled_seqs

        fig = px.scatter_3d(
            df, x='PC1', y='PC2', z='PC3',
            color='Sample',  # Each sample gets unique color
            hover_data=['Sample'],
            title='Interactive 3D PCA of Protein Embeddings (10 Samples)',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            }
        )
        fig.write_html('3d_pca_interactive_10samples.html')
        print(f"Saved interactive 3D PCA visualization to 3d_pca_interactive_10samples.html")
    except ImportError:
        print("Plotly not available. Only static plot created.")

    plt.show()
    return pca_result, sampled_keys


def plot_2d_TSNE(data, keys, seqs=None, num_samples=100, perplexity=30, learning_rate=200):
    """
    Plot 2D t-SNE visualization of the embeddings.

    Args:
        data: List of embeddings with shape (36, 1152)
        keys: List of identifiers for each embedding
        seqs: Optional list of sequences for annotation
        num_samples: Number of samples to plot (default: 100)
        perplexity: t-SNE perplexity parameter (default: 30)
        learning_rate: t-SNE learning rate (default: 200)
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np
    import random

    # Limit to num_samples if there are more samples
    if len(data) > num_samples:
        # Select random indices
        indices = random.sample(range(len(data)), num_samples)
        sampled_data = [data[i] for i in indices]
        sampled_keys = [keys[i] for i in indices]
        sampled_seqs = [seqs[i] for i in indices] if seqs is not None else None
    else:
        sampled_data = data
        sampled_keys = keys
        sampled_seqs = seqs

    # Convert embeddings to feature vectors
    feature_vectors = []
    for embedding in sampled_data:
        # Mean across features to get (36,)
        feature_vectors.append(embedding.mean(axis=1))

    feature_vectors = np.array(feature_vectors)

    # Apply t-SNE to reduce to 2 dimensions
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(feature_vectors)-1),
                learning_rate=learning_rate, random_state=42)
    tsne_result = tsne.fit_transform(feature_vectors)

    # Create 2D plot with larger size
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111)

    # Use a colormap with distinct colors for each sample
    cmap = plt.cm.get_cmap('tab10', num_samples)

    scatter = ax.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=range(len(tsne_result)),  # Each sample gets its own color
        cmap=cmap,
        s=100,  # Larger dot size for 2D plot
        alpha=0.8
    )

    # Add annotations for all points
    for i, key in enumerate(sampled_keys):
        ax.text(tsne_result[i, 0], tsne_result[i, 1], key, fontsize=9)

    # Set labels and title
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title(f'2D t-SNE of Protein Embeddings ({num_samples} Samples)', fontsize=16)

    # Add a color bar with sample IDs
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, ticks=range(len(sampled_keys)))
    cbar.set_label('Sample ID')
    cbar.ax.set_yticklabels(sampled_keys)

    plt.tight_layout()
    plt.savefig('2d_tsne_embeddings.png', dpi=300)
    print(f"Saved 2D t-SNE visualization to 2d_tsne_embeddings.png")

    # Create interactive plot if plotly is available
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame({
            'Dimension 1': tsne_result[:, 0],
            'Dimension 2': tsne_result[:, 1],
            'Sample': sampled_keys
        })

        if sampled_seqs is not None:
            df['Sequence'] = sampled_seqs

        fig = px.scatter(
            df, x='Dimension 1', y='Dimension 2',
            color='Sample',  # Each sample gets unique color
            hover_data=['Sample'],
            title=f'Interactive 2D t-SNE of Protein Embeddings ({num_samples} Samples)',
        )
        fig.write_html('2d_tsne_interactive.html')
        print(f"Saved interactive 2D t-SNE visualization to 2d_tsne_interactive.html")
    except ImportError:
        print("Plotly not available. Only static plot created.")

    plt.show()
    return tsne_result, sampled_keys


# def plot_(data, keys):
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE
#     from sklearn.cluster import KMeans
#     import seaborn as sns
#     import numpy as np
#     from scipy.spatial.distance import pdist, squareform
#
#     for arr, key in zip(data, keys):
#         try:
#             if len(arr.shape) != 2:
#                 print(f"Skipping {key} as it's not 2D")
#                 continue
#
#             n_samples, n_features = arr.shape
#             print(f"Visualizing {key}: {n_samples} samples × {n_features} features")
#
#             fig = plt.figure(figsize=(18, 15))
#
#             # 1. PCA visualization
#             ax1 = plt.subplot(2, 2, 1)
#             pca = PCA(n_components=2)
#             pca_result = pca.fit_transform(arr)
#
#             # Apply KMeans to color points by cluster
#             n_clusters = min(5, n_samples)
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             clusters = kmeans.fit_predict(arr)
#
#             scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
#                         c=clusters, cmap='viridis', alpha=0.8, s=100)
#             ax1.set_title(f'PCA Projection')
#             ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
#             ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
#             plt.colorbar(scatter, ax=ax1, label='Cluster')
#
#             # 2. t-SNE visualization
#             ax2 = plt.subplot(2, 2, 2)
#             perplexity = min(30, max(5, n_samples-1))
#             tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
#             tsne_result = tsne.fit_transform(arr)
#
#             scatter2 = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1],
#                          c=clusters, cmap='viridis', alpha=0.8, s=100)
#             ax2.set_title(f't-SNE Projection')
#             plt.colorbar(scatter2, ax=ax2, label='Cluster')
#
#             # 3. Sequence similarity heatmap
#             ax3 = plt.subplot(2, 2, 3)
#             distances = squareform(pdist(arr, metric='euclidean'))
#             sns.heatmap(distances, cmap='coolwarm', ax=ax3)
#             ax3.set_title('Pairwise Distance Matrix')
#             ax3.set_xlabel('Sequence Index')
#             ax3.set_ylabel('Sequence Index')
#
#             # 4. Feature variance visualization
#             ax4 = plt.subplot(2, 2, 4)
#             feature_variance = np.var(arr, axis=0)
#             sorted_idx = np.argsort(feature_variance)[::-1]
#             top_k = 50  # Show top 50 most variable features
#             ax4.bar(range(top_k), feature_variance[sorted_idx[:top_k]])
#             ax4.set_title('Top Variable Features')
#             ax4.set_xlabel('Feature Rank')
#             ax4.set_ylabel('Variance')
#
#             plt.suptitle(f'Visualization of {key} Embeddings', fontsize=16)
#             plt.tight_layout(rect=[0, 0, 1, 0.96])
#
#             plt.savefig(f'visualization_{key}.png', dpi=300)
#             print(f"Saved visualization to visualization_{key}.png")
#             plt.close()
#
#         except Exception as e:
#             print(f"Error visualizing {key}: {str(e)}")

def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    directory = current_path
    data, keys, seqs = load_npz_files(directory)
    print_shapes(data, keys, seqs)
    ## print_samples(data, keys)
    ## plot_(data, keys)
    # selected_keys = [('HLA-B3813', 'HLA-B3814'), ('HLA-B38:01', 'HLA-B38:05')]  # second sample has same pseudoseq
    selected_keys = [('HLA-B13020103', 'HLA-B13020105')]
    plot_pairwise_comparison(selected_keys, keys, data, seqs)
    plot_3d_PCA(data, keys, seqs)
    plot_2d_TSNE(data, keys, seqs)

if __name__ == "__main__":
    main()