
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils.model import SCQ1DAutoEncoder
'''
# Set TensorFlow logging level for more information
tf.get_logger().setLevel('INFO')


# def create_dataset(X, batch_size=1, is_training=True):
#     """Create TensorFlow dataset with consistent approach for both training and validation."""
#     # Debug info
#     print(f"Creating {'training' if is_training else 'validation'} dataset")
#     print(f"Data shape: {X.shape}")
#     print(f"Data dtype: {X.dtype}")
#     print(f"Data range: min={np.min(X):.4f}, max={np.max(X):.4f}, mean={np.mean(X):.4f}")
#
#     # Ensure data is float32 (TensorFlow works best with float32)
#     X = X.astype(np.float32)
#
#     dataset = tf.data.Dataset.from_tensor_slices(X)
#
#     # Apply shuffling only for training data
#     if is_training:
#         dataset = dataset.shuffle(buffer_size=1000)
#
#     # Apply batching with drop_remainder=False to handle all data
#     dataset = dataset.batch(batch_size)
#
#     # Prefetch for better performance
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
#
#     # Debug dataset
#     for batch in dataset.take(1):
#         print(f"Sample batch shape: {batch.shape}")
#         print(f"Sample batch dtype: {batch.dtype}")
#         print(f"Sample batch range: min={tf.reduce_min(batch):.4f}, max={tf.reduce_max(batch):.4f}")
#
#     return dataset

def create_dataset(X, batch_size=1, is_training=True):
    """
    simple function to create a TensorFlow dataset.
    Args:
        X:
        batch_size:
        is_training:

    Returns:

    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)

    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices(X)

    # Apply shuffling only for training data
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(1000, len(X)))

    # Apply batching (drop_remainder=False to handle all data)
    dataset = dataset.batch(batch_size)

    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def plot_metrics(history, save_path='training_metrics.png'):
            """
            Visualize training metrics over epochs.

            Args:
                history: A Keras History object or dictionary containing training history
                save_path: Path to save the plot image
            """
            # Handle both Keras History objects and dictionaries
            history_dict = history.history if hasattr(history, 'history') else history

            # Print available keys to debug
            print(f"Available keys in history: {list(history_dict.keys())}")

            metrics = ['loss', 'recon', 'vq', 'perplexity']
            # Check for standard Keras naming (total_loss instead of loss, etc.)
            metric_mapping = {
                'loss': ['loss', 'total_loss'],
                'recon': ['recon', 'recon_loss'],
                'vq': ['vq', 'vq_loss'],
                'perplexity': ['perplexity']
            }

            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3*len(metrics)), sharex=True)

            # Handle case with only one metric
            if len(metrics) == 1:
                axes = [axes]

            for i, metric_base in enumerate(metrics):
                ax = axes[i]

                # Try different possible metric names
                for metric in metric_mapping[metric_base]:
                    # Plot training metric
                    if metric in history_dict:
                        ax.plot(history_dict[metric], 'b-', label=f'Train {metric}')
                        print(f"Plotting {metric} with {len(history_dict[metric])} points")
                        break

                # Try different possible validation metric names
                for metric in metric_mapping[metric_base]:
                    val_metric = f'val_{metric}'
                    if val_metric in history_dict:
                        ax.plot(history_dict[val_metric], 'r-', label=f'Validation {metric}')
                        print(f"Plotting {val_metric} with {len(history_dict[val_metric])} points")
                        break

                ax.set_title(f'{metric_base.capitalize()} over epochs')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend(loc='best')

            plt.xlabel('Epochs')
            plt.tight_layout()

            # Save the figure in case display doesn't work
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

            # Try to display
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")


def main():
    print("Starting SCQ parameter search...")

    try:
        print("Loading peptide embeddings...")
        mhc1_pep2vec_embeddings = pd.read_parquet("data/Pep2Vec/wrapper_mhc1.parquet")
        mhc2_pep2vec_embeddings = pd.read_parquet("data/Pep2Vec/wrapper_mhc2.parquet")

        # Select the latent columns (the columns that has latent in their name)
        mhc1_latent_columns = [col for col in mhc1_pep2vec_embeddings.columns if 'latent' in col]
        mhc2_latent_columns = [col for col in mhc2_pep2vec_embeddings.columns if 'latent' in col]

        print(f"Found {len(mhc1_latent_columns)} latent columns for MHC1")
        print(f"Found {len(mhc2_latent_columns)} latent columns for MHC2")

        if len(mhc1_latent_columns) == 0 or len(mhc2_latent_columns) == 0:
            print("WARNING: No latent columns found. Check column names.")

        # Extract latent features
        X_mhc1 = mhc1_pep2vec_embeddings[mhc1_latent_columns].values
        X_mhc2 = mhc2_pep2vec_embeddings[mhc2_latent_columns].values

        print(f"MHC1 data shape: {X_mhc1.shape}")
        print(f"MHC2 data shape: {X_mhc2.shape}")

        # Data sanity check
        print("Data overview:")
        print(
            f"MHC1 - min: {np.min(X_mhc1):.4f}, max: {np.max(X_mhc1):.4f}, mean: {np.mean(X_mhc1):.4f}, std: {np.std(X_mhc1):.4f}")
        print(
            f"MHC2 - min: {np.min(X_mhc2):.4f}, max: {np.max(X_mhc2):.4f}, mean: {np.mean(X_mhc2):.4f}, std: {np.std(X_mhc2):.4f}")

        # Check for NaN or infinity values
        print(f"MHC1 has NaN: {np.isnan(X_mhc1).any()}, has inf: {np.isinf(X_mhc1).any()}")
        print(f"MHC2 has NaN: {np.isnan(X_mhc2).any()}, has inf: {np.isinf(X_mhc2).any()}")

        # Replace any NaN values with zeros
        if np.isnan(X_mhc1).any():
            print("Replacing NaN values in MHC1 with zeros")
            X_mhc1 = np.nan_to_num(X_mhc1)

        if np.isnan(X_mhc2).any():
            print("Replacing NaN values in MHC2 with zeros")
            X_mhc2 = np.nan_to_num(X_mhc2)

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create output directory for results
    output_dir = "output/scq_parameter_search"
    os.makedirs(output_dir, exist_ok=True)

    # Define parameter search space - simplified to just one configuration for testing
    param_grid = [
        # Explore different codebook_num
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 8, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 32, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 64, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 128, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 256, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 512, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 1024, 'heads': 4},
        # # Explore codebook_dim
        # {'general_embed_dim': 128, 'codebook_dim': 32, 'codebook_num': 16, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 64, 'codebook_num': 16, 'heads': 4},
        # {'general_embed_dim': 128, 'codebook_dim': 128, 'codebook_num': 16, 'heads': 4},
        # # Explore heads
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 2},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 8},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 16},
        # {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 32},
        # # Explore general_embed_dim
        # {'general_embed_dim': 64, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
        # {'general_embed_dim': 256, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
        # {'general_embed_dim': 512, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
    ]

    # Common parameters for all configurations
    common_params = {
        'descrete_loss': True,
        'weight_recon': 1.0,
        'weight_vq': 1.0,
    }

    # Set up cross-validation - reduced to 2 folds for testing
    n_folds = 2
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Batch size for all datasets
    batch_size = 1

    # Store results
    results = []

    # Run parameter search for each dataset
    # Just use MHC1 for testing
    for dataset_name, X, latent_columns in [
        ("MHC1", X_mhc1, mhc1_latent_columns),
        ("MHC2", X_mhc2, mhc2_latent_columns)
    ]:
        print(f"\n{'=' * 50}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'=' * 50}\n")

        # Create directory for this dataset
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Model selection loop
        for param_idx, param_config in enumerate(param_grid):
            config_name = f"config_{param_idx + 1}"
            print(f"\n{'-' * 50}")
            print(f"Testing {config_name}: {param_config}")
            print(f"{'-' * 50}\n")

            # Merge with common parameters
            model_params = {**common_params, **param_config}

            # Create config directory
            config_dir = os.path.join(dataset_dir, config_name)
            os.makedirs(config_dir, exist_ok=True)

            # Cross-validation metrics
            cv_metrics = {
                'loss': [],
                'recon': [],
                'vq': [],
                'perplexity': []
            }

            # Prepare folds
            fold_indices = list(kf.split(X))

            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
                print(f"Training fold {fold_idx + 1}/{n_folds}")

                # Split data
                X_train, X_val = X[train_idx], X[val_idx]

                # Check data shapes
                print(f"Train data shape: {X_train.shape}")
                print(f"Validation data shape: {X_val.shape}")

                # Convert to TensorFlow dataset using the new function
                train_dataset = create_dataset(X_train, batch_size=batch_size, is_training=True)
                val_dataset = create_dataset(X_val, batch_size=batch_size, is_training=False)

                # Define the model
                model = SCQ_model(**model_params, input_dim=X_train.shape[1])

                # Print model summary
                model.build((None, X_train.shape[1]))
                print("Model summary:")
                model.summary()

                # Compile the model
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

                # Prepare callbacks
                fold_dir = os.path.join(config_dir, f"fold_{fold_idx}")
                os.makedirs(fold_dir, exist_ok=True)

                # Create a custom callback to print metrics after each epoch
                class MetricsPrinter(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        print(f"\nEpoch {epoch + 1} metrics:")
                        for metric, value in logs.items():
                            print(f"  {metric}: {value:.6f}")

                callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(fold_dir, 'checkpoint'),
                        save_best_only=True,
                        monitor='val_total_loss',
                        mode='min'
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_total_loss',
                        patience=5,
                        restore_best_weights=True
                    ),
                    MetricsPrinter()  # Add the custom callback
                ]

                # Train the model with fewer epochs for testing
                print("Starting model training...")
                try:
                    history = model.fit(
                        train_dataset,
                        epochs=30,  # Reduced for testing
                        validation_data=val_dataset,
                        callbacks=callbacks,
                        verbose=1
                    )

                    # Check if history contains expected metrics
                    print("\nHistory keys:", history.history.keys())
                    for key, values in history.history.items():
                        print(f"{key}: {values}")

                    # plot
                    plot_metrics(history)

                except Exception as e:
                    print(f"Error during training: {e}")
                    continue

                # Save training history
                try:
                    with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
                        json.dump(history.history, f)
                except Exception as e:
                    print(f"Error saving history: {e}")

                # Reset metrics before evaluation
                print("Resetting metrics before evaluation")
                model.reset_metrics()

                # Evaluate on validation set
                print("Evaluating model on validation data...")
                # After getting validation metrics
                try:
                    val_metrics = model.evaluate(val_dataset, return_dict=True)
                    print("Validation metrics:", val_metrics)

                    # Map the metric names
                    metric_mapping = {
                        'loss': 'total_loss',
                        'recon': 'recon_loss',
                        'vq': 'vq_loss',
                        'perplexity': 'perplexity'
                    }

                    # Record metrics with mapping
                    for returned_name, expected_name in metric_mapping.items():
                        if returned_name in val_metrics:
                            cv_metrics[expected_name].append(val_metrics[returned_name])
                        else:
                            cv_metrics[expected_name].append(0)

                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    # Rest of your error handling

                # Record metrics
                for metric in cv_metrics:
                    cv_metrics[metric].append(val_metrics[metric])

                # Save fold metrics
                with open(os.path.join(fold_dir, 'metrics.json'), 'w') as f:
                    json.dump(val_metrics, f, indent=2)

            # Calculate average metrics across folds
            avg_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
            std_metrics = {k: np.std(v) for k, v in cv_metrics.items()}

            # Report results
            print(f"\nAverage metrics for {config_name}:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.6f} (±{std_metrics[metric]:.6f})")

            # Store results
            result = {
                'dataset': dataset_name,
                'config': config_name,
                **model_params,
                **avg_metrics,
                **{f"{k}_std": v for k, v in std_metrics.items()},
                'cv_metrics': cv_metrics
            }
            results.append(result)

            # Save config results
            with open(os.path.join(config_dir, 'results.json'), 'w') as f:
                json.dump(result, f, indent=2)

    # Save all results to CSV
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'cv_metrics'} for r in results])
    results_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

    print("\nParameter search complete!")
    print(f"Results saved to {output_dir}")

import os
import traceback

def simple_run(batch_size=1):
    """A simplified run using the MHC dataset instead of random data."""
    print("Starting simplified SCQ model run on MHC data...")

    try:
        # Create output directory for results
        output_dir = "output/scq_simple_run"
        os.makedirs(output_dir, exist_ok=True)

        # Load MHC data
        print("Loading peptide embeddings...")
        mhc1_pep2vec_embeddings = pd.read_parquet("data/Pep2Vec/wrapper_mhc1.parquet")

        # Select the latent columns
        latent_columns = [col for col in mhc1_pep2vec_embeddings.columns if 'latent' in col]
        print(f"Found {len(latent_columns)} latent columns for MHC1")

        # Extract latent features
        X = mhc1_pep2vec_embeddings[latent_columns].values
        print(f"MHC1 data shape: {X.shape}")

        # Data sanity check
        print("Data overview:")
        print(f"MHC1 - min: {np.min(X):.4f}, max: {np.max(X):.4f}, mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")

        # Replace any NaN values with zeros
        if np.isnan(X).any():
            print("Replacing NaN values in MHC1 with zeros")
            X = np.nan_to_num(X)

        # Split data into train and test sets (80/20 split)
        from sklearn.model_selection import train_test_split
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        # Create datasets
        train_dataset = create_dataset(X_train, batch_size=batch_size, is_training=True)
        test_dataset = create_dataset(X_test, batch_size=batch_size, is_training=False)

        # Initialize model with dimensions matching the input data
        model = SCQ_model(
            general_embed_dim=128,
            codebook_dim=16,
            codebook_num=8,
            descrete_loss=True,
            heads=4,
            input_dim=X_train.shape[1]
        )

        # Print model summary
        model.build(input_shape=(None, X_train.shape[1]))
        print("Model summary:")
        model.summary()

        # Compile model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

        # Train with early stopping
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'model_checkpoint.weights.h5'),
                save_best_only=True,
                save_weights_only=True,  # Only save weights, not the full model
                monitor='loss'
            )
        ]

        # Train model and capture history
        history = model.fit(
            train_dataset,
            epochs=1000,  # Reduced from 1000 for quicker testing
            callbacks=callbacks,
            verbose=1
        )

        # Plot training metrics
        plot_metrics(history, save_path=os.path.join(output_dir, 'training_metrics.png'))

        # Save training history
        pd.DataFrame(history.history).to_csv(os.path.join(output_dir, "model_history.csv"), index=False)

        # Test model on test data
        print("Evaluating model on test data...")
        evaluation = model.evaluate(test_dataset, return_dict=True)
        print("Test metrics:", evaluation)

        # Generate and save example outputs
        for batch in test_dataset.take(1):
            output = model(batch)
            # Save sample input and output
            np.save(os.path.join(output_dir, "sample_input.npy"), batch.numpy())
            np.savez(
                os.path.join(output_dir, "sample_output.npz"),
                decoded=output[0].numpy(),
                zq=output[1].numpy(),
                pj=output[2].numpy()
            )

        print(f"Training complete. Results saved to {output_dir}")

    except Exception as e:
        print(f"Error in simple_run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # try:
    #     main()
    # except Exception as e:
    #     print(f"Error in main function: {e}")
    #     import traceback
    #
    #     traceback.print_exc()
    # simple run
    simple_run()'''
from sklearn.model_selection import train_test_split

'''import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools
from utils.model import SCQ_model


def create_dataset(X, batch_size=4, is_training=True):
    """Create a TensorFlow dataset from input array X."""
    dataset = tf.data.Dataset.from_tensor_slices(X)
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_data(data_path, test_size=0.2, random_state=42):
    """Load and prepare data for training."""
    # Load data
    embeddings = pd.read_parquet(data_path)

    # Select the latent columns
    latent_columns = [col for col in embeddings.columns if 'latent' in col]
    print(f"Found {len(latent_columns)} latent columns")

    # Extract values and reshape
    X = embeddings[latent_columns].values
    seq_length = len(latent_columns)
    feature_dim = 1
    X = X.reshape(-1, seq_length, feature_dim)

    # Split into train and test sets
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    return X_train, X_test, seq_length, feature_dim


def train_scq_model(X_train, X_test, feature_dim, general_embed_dim, codebook_dim,
                    codebook_num, batch_size=4, epochs=20, learning_rate=0.001,
                    heads=8, descrete_loss=False, output_dir="test_tmp"):
    """Train an SCQ model with the given parameters."""
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create datasets
    train_dataset = create_dataset(X_train, batch_size=batch_size, is_training=True)
    test_dataset = create_dataset(X_test, batch_size=batch_size, is_training=False)

    # Initialize SCQ model
    model = SCQ_model(input_dim=int(feature_dim),
                      general_embed_dim=int(general_embed_dim),
                      codebook_dim=int(codebook_dim),
                      codebook_num=int(codebook_num),
                      descrete_loss=descrete_loss,
                      heads=int(heads))

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

    # Train model and capture history
    history = model.fit(train_dataset, epochs=epochs, batch_size=batch_size)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(output_dir, f"model_history_cb{codebook_num}_emb{general_embed_dim}.csv")
    history_df.to_csv(history_path, index=False)

    # Evaluate model on test data
    decoded_outputs, zq_outputs, pj_outputs = evaluate_model(model, test_dataset)

    # Save outputs
    output_path = os.path.join(output_dir, f"output_data_cb{codebook_num}_emb{general_embed_dim}.npz")
    np.savez(output_path,
             decoded=np.vstack(decoded_outputs),
             zq=np.vstack(zq_outputs),
             pj=np.vstack(pj_outputs))

    # Calculate metrics
    mse = calculate_reconstruction_mse(X_test, np.vstack(decoded_outputs))

    return model, history, mse


def evaluate_model(model, test_dataset):
    """Evaluate the model on test data."""
    decoded_outputs = []
    zq_outputs = []
    pj_outputs = []

    for batch in test_dataset:
        output = model(batch)
        decoded_outputs.append(output[0].numpy())
        zq_outputs.append(output[1].numpy())
        pj_outputs.append(output[2].numpy())

    return decoded_outputs, zq_outputs, pj_outputs


def calculate_reconstruction_mse(X_test, decoded_output):
    """Calculate mean squared error between input and reconstructed output."""
    # Reshape if necessary to match dimensions
    if X_test.shape != decoded_output.shape:
        # Adjust shapes as needed based on your model's output
        pass

    return mean_squared_error(X_test.reshape(-1), decoded_output.reshape(-1))


def plot_training_history(history_df, title="Training History", output_path=None):
    """Plot training metrics from history dataframe."""
    plt.figure(figsize=(12, 6))

    # Plot all metrics in the history
    for column in history_df.columns:
        plt.plot(history_df[column], label=column)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_reconstruction_comparison(original, reconstructed, n_samples=5, output_path=None):
    """Plot comparison between original and reconstructed sequences."""
    # Randomly select n_samples sequences to visualize
    indices = np.random.choice(len(original), size=min(n_samples, len(original)), replace=False)

    plt.figure(figsize=(15, 3 * n_samples))

    for i, idx in enumerate(indices):
        # Plot original sequence
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.plot(original[idx].flatten())
        plt.title(f"Original Sequence {idx}")
        plt.grid(True)

        # Plot reconstructed sequence
        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.plot(reconstructed[idx].flatten())
        plt.title(f"Reconstructed Sequence {idx}")
        plt.grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def parameter_search(X_train, X_test, feature_dim, batch_size=4, epochs=20, learning_rate=0.001,
                     codebook_nums=[4,8,16,32,64,128,256,512,1024], embed_dims=[64, 128, 256],
                     codebook_dim=21, heads=8, output_dir="parameter_search"):
    """
    Perform grid search over codebook_num and general_embed_dim parameters.
    Returns the best parameters based on reconstruction MSE.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # Create all combinations of parameters
    param_combinations = list(itertools.product(codebook_nums, embed_dims))
    total_combinations = len(param_combinations)

    print(f"Starting parameter search with {total_combinations} combinations...")

    for i, (codebook_num, embed_dim) in enumerate(param_combinations):
        print(f"Training combination {i + 1}/{total_combinations}: codebook_num={codebook_num}, embed_dim={embed_dim}")

        try:
            # Train the model with this parameter combination
            model, history, mse = train_scq_model(
                X_train, X_test, feature_dim,
                general_embed_dim=embed_dim,
                codebook_dim=codebook_dim,
                codebook_num=codebook_num,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                heads=heads,
                output_dir=output_dir
            )

            # Store results
            results.append({
                'codebook_num': codebook_num,
                'general_embed_dim': embed_dim,
                'mse': mse,
                'history': history
            })

            # Plot training history
            history_df = pd.DataFrame(history.history)
            plot_training_history(
                history_df,
                title=f"Training History (codebook_num={codebook_num}, embed_dim={embed_dim})",
                output_path=os.path.join(output_dir, f"history_plot_cb{codebook_num}_emb{embed_dim}.png")
            )

        except Exception as e:
            print(f"Error training with codebook_num={codebook_num}, embed_dim={embed_dim}: {e}")

    # Create results dataframe
    results_df = pd.DataFrame([(r['codebook_num'], r['general_embed_dim'], r['mse'])
                               for r in results],
                              columns=['codebook_num', 'general_embed_dim', 'mse'])

    # Save results
    results_df.to_csv(os.path.join(output_dir, "parameter_search_results.csv"), index=False)

    # Find best parameters
    best_idx = results_df['mse'].idxmin()
    best_params = results_df.loc[best_idx]

    print(f"Parameter search complete. Best parameters:")
    print(f"  codebook_num: {best_params['codebook_num']}")
    print(f"  general_embed_dim: {best_params['general_embed_dim']}")
    print(f"  MSE: {best_params['mse']}")

    # Plot results heatmap
    plot_parameter_search_results(results_df, output_dir)

    return best_params, results_df


def plot_parameter_search_results(results_df, output_dir):
    """Plot heatmap of parameter search results."""
    # Create pivot table for heatmap
    pivot_df = results_df.pivot(index='codebook_num', columns='general_embed_dim', values='mse')

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_df, cmap='viridis_r')  # Reverse colormap so darker is better (lower MSE)

    # Set labels
    plt.colorbar(label='MSE (lower is better)')
    plt.title('Parameter Search Results')
    plt.xlabel('General Embedding Dimension')
    plt.ylabel('Codebook Number')

    # Set tick labels
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)

    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                plt.text(j, i, f'{value:.4f}', ha='center', va='center',
                         color='white' if value > pivot_df.values.mean() else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_search_heatmap.png"))
    plt.close()


def run_scq_pipeline(data_path, output_dir="test_tmp", run_param_search=True,
                     batch_size=4, epochs=3, learning_rate=0.001,
                     codebook_num=5, general_embed_dim=128, codebook_dim=32, heads=8):
    """
    Main function to run the complete SCQ pipeline with optional parameter search.
    """
    # Load and prepare data
    X_train, X_test, seq_length, feature_dim = load_data(data_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save input data
    np.save(os.path.join(output_dir, "input_data.npy"), X_test)

    if run_param_search:
        # Run parameter search to find optimal values
        best_params, results_df = parameter_search(
            X_train, X_test, feature_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            output_dir=os.path.join(output_dir, "param_search")
        )

        # Use best parameters for final model - ensure they are integers
        codebook_num = int(best_params['codebook_num'])
        general_embed_dim = int(best_params['general_embed_dim'])

    # Train final model with selected/best parameters
    print(f"Training final model with codebook_num={codebook_num}, general_embed_dim={general_embed_dim}")
    final_model, final_history, final_mse = train_scq_model(
        X_train, X_test, feature_dim,
        general_embed_dim=general_embed_dim,
        codebook_dim=codebook_dim,
        codebook_num=codebook_num,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        heads=heads,
        output_dir=output_dir
    )

    # Evaluate the final model
    test_dataset = create_dataset(X_test, batch_size=batch_size, is_training=False)
    decoded_outputs, zq_outputs, pj_outputs = evaluate_model(final_model, test_dataset)

    # Create visualizations
    history_df = pd.DataFrame(final_history.history)
    plot_training_history(
        history_df,
        title=f"Final Model Training History (codebook_num={codebook_num}, embed_dim={general_embed_dim})",
        output_path=os.path.join(output_dir, "final_model_history_plot.png")
    )

    plot_reconstruction_comparison(
        X_test,
        np.vstack(decoded_outputs),
        n_samples=5,
        output_path=os.path.join(output_dir, "reconstruction_comparison.png")
    )

    print(f"Pipeline completed successfully.")
    print(f"Final model MSE: {final_mse}")
    print(f"Final model parameters: codebook_num={codebook_num}, general_embed_dim={general_embed_dim}")
    print(f"Results saved to {output_dir}")

    return final_model, final_mse, (codebook_num, general_embed_dim)


# Example usage:
if __name__ == "__main__":
    # Example usage of the pipeline
    data_path = "data/Pep2Vec/wrapper_mhc1.parquet"

    # # Run full pipeline with parameter search
    # model, mse, best_params = run_scq_pipeline(
    #     data_path=data_path,
    #     output_dir="scq_results",
    #     run_param_search=True,
    #     batch_size=4,
    #     epochs=20
    # )

    # Alternatively, run with specific parameters (no search)
    model, mse, params = run_scq_pipeline(
        data_path=data_path,
        output_dir="scq_results_fixed",
        run_param_search=False,
        codebook_num=16,
        codebook_dim=64,
        heads=8,
        batch_size=1,
        general_embed_dim=256,
        epochs=20
    )'''


# TODO implement a simple training pipeline for VQUnet
def train_and_evaluate_scqvae(
    data_path,
    num_embeddings=32,
    embedding_dim=64,  # Increased from 32
    batch_size=32,      # Increased from 1
    epochs=20,         # Increased from 10
    learning_rate=1e-5,  # Adjusted from 1e-4
    commitment_beta=0.25,
    output_dir='data/SCQvae',
    visualize=True,
    save_model=True,
    random_state=42,
    test_size=0.2,
    output_data="all", # Options: "val", "train", "all"
    **kwargs
):
    """
    Train and evaluate a VQ-VAE model on peptide embedding data with improved codebook utilization.

    Args:
        data_path: Path to the parquet file containing peptide embeddings
        num_embeddings: Number of clusters/codes in the codebook
        embedding_dim: Dimension of each codebook vector
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        commitment_beta: Beta parameter for commitment loss
        output_dir: Directory to save outputs
        visualize: Whether to generate visualizations
        save_model: Whether to save model weights
        random_state: Random seed for reproducibility
        test_size: Fraction of data to use for validation

    Returns:
        model: Trained model
        history: Training history
        latent_data: Dictionary containing latent representations and indices
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Helper Functions ---
    def create_dataset(X, batch_size=32, is_training=True):
        """Create a TensorFlow dataset from input array X."""
        # X shape: (num_samples, seq_length)
        dataset = tf.data.Dataset.from_tensor_slices(X)
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(X))  # Shuffle all samples
        dataset = dataset.batch(batch_size)  # Result shape: (batch_size, seq_length)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch for better performance
        return dataset

    def load_data(data_path, test_size=0.2, random_state=42):
        """Load and prepare data for training."""
        embeddings = pd.read_parquet(data_path)
        latent_columns = [col for col in embeddings.columns if 'latent' in col]
        print(f"Found {len(latent_columns)} latent columns")
        X = embeddings[latent_columns].values
        seq_length = len(latent_columns)
        X = X.reshape(-1, seq_length)
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        return X_train, X_test, seq_length

    def initialize_codebook_with_kmeans(X_train, num_embeddings, embedding_dim):
        """Initialize codebook vectors using k-means clustering."""
        kmeans = KMeans(n_clusters=num_embeddings, random_state=random_state)
        flat_data = X_train.reshape(-1, X_train.shape[-1])
        kmeans.fit(flat_data)
        return kmeans.cluster_centers_.astype(np.float32)

    def plot_training_metrics(history, save_path=None):
        """Plot the training metrics over epochs."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
        axes[0].plot(history.history['total_loss'], 'b-', label='Train')
        if 'val_total_loss' in history.history:
            axes[0].plot(history.history['val_total_loss'], 'r-', label='Validation')
        axes[0].set_title('Total Loss')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(history.history['recon_loss'], 'b-', label='Train')
        if 'val_recon_loss' in history.history:
            axes[1].plot(history.history['val_recon_loss'], 'r-', label='Validation')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        axes[1].legend()

        axes[2].plot(history.history['vq_loss'], 'b-', label='Train')
        if 'val_vq_loss' in history.history:
            axes[2].plot(history.history['val_vq_loss'], 'r-', label='Validation')
        axes[2].set_title('VQ Loss')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        axes[2].legend()

        axes[3].plot(history.history['perplexity'], 'b-', label='Train')
        if 'val_perplexity' in history.history:
            axes[3].plot(history.history['val_perplexity'], 'r-', label='Validation')
        axes[3].set_title('Perplexity')
        axes[3].set_xlabel('Epochs')
        axes[3].set_ylabel('Perplexity')
        axes[3].grid(True)
        axes[3].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if visualize:
            plt.show()
        else:
            plt.close()

    def plot_reconstructions(original, reconstructed, n_samples=5, save_path=None):
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

    def plot_codebook_usage(indices, num_embeddings, save_path=None):
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
        cbar = plt.colorbar(sm)
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

    def plot_soft_cluster_distribution(soft_cluster_probs, num_samples=None, save_path=None):
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
                 f"Active clusters: {active_clusters}/{n_clusters} ({active_clusters/n_clusters:.1%})\n"
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


    def plot_cluster_distribution(soft_cluster_probs, save_path=None):
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
        cbar = plt.colorbar(sm)
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

    # --- Main Pipeline ---
    print("Loading/Generating Training Data...")
    train_data, test_data, seq_length = load_data(data_path, test_size=test_size, random_state=random_state)

    # Initialize codebook with k-means
    print("Initializing codebook with k-means...")
    initial_codebook = initialize_codebook_with_kmeans(train_data, num_embeddings, seq_length)

    # Create datasets
    print("Creating TensorFlow Datasets...")
    train_dataset = create_dataset(train_data, batch_size=batch_size, is_training=True)
    val_dataset = create_dataset(test_data, batch_size=batch_size, is_training=False)
    print("Datasets created.")

    # --- Model Instantiation ---
    print("Building the SCQ1DAutoEncoder model...")
    input_shape = (seq_length,)
    model = SCQ1DAutoEncoder(
        input_dim=input_shape,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_beta=commitment_beta,
        scq_params={
            'lambda_reg': 1.0,          # Increased regularization
            'discrete_loss': True,      # Use discrete loss
            'reset_dead_codes': True,   # Reset underused vectors
            'usage_threshold': 1e-4,    # Lower threshold
            'reset_interval': 5         # Frequent resets
        },
        initial_codebook=initial_codebook  # Pass k-means initialized codebook
    )
    print("Model built.")

    # --- Compile and Train ---
    print("Compiling the model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    print("Model compiled.")

    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset
    )
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    # --- Evaluation and Visualization ---
    if visualize:
        print("\nPlotting training history...")
        plot_training_metrics(history, save_path=os.path.join(output_dir, 'vqvae_training_metrics.png'))
        print("\nPerforming example inference...")
        example_batch = next(iter(val_dataset))
        output = model(example_batch, training=False)
        reconstruction, quantized_latent, cluster_indices, vq_loss, perplexity = output
        print("\nVisualizing reconstructions...")
        plot_reconstructions(example_batch.numpy(), reconstruction.numpy(),
                             save_path=os.path.join(output_dir, 'vqvae_reconstructions.png'))
        plot_reconstructions(example_batch.numpy(), reconstruction.numpy(),
                             save_path=os.path.join(output_dir, 'vqvae_reconstructions.png'))
        print("\nAnalyzing codebook usage...")
        # # Use hard indices for codebook usage analysis (index 5 in model output)
        # plot_codebook_usage(cluster_indices, num_embeddings, save_path=os.path.join(output_dir, 'codebook_usage.png'))
        # print("\nAnalyzing cluster distribution from one-hot encodings...")
        # # Use one-hot encodings for cluster distribution (index 2 in model output)
        # # print number of samples
        # print(f"Shape of cluster_indices: {cluster_indices.shape}")
        # plot_cluster_distribution(cluster_indices, save_path=os.path.join(output_dir, 'cluster_distribution.png'))

    # --- Save Model ---
    if save_model:
        model_dir = os.path.join(output_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model.save_weights(os.path.join(model_dir, 'vqvae_model_weights.h5'))
        print(f"\nModel weights saved to '{os.path.join(model_dir, 'vqvae_model_weights.h5')}'")

    # --- Extract Latent Space ---
    print("\nExtracting quantized latent space...")
    if output_data == "all":
        out_dataset = tf.data.Dataset.concatenate(train_dataset, val_dataset)
    elif output_data == "train":
        out_dataset = train_dataset # using only training dataset for quantization
    else:
        out_dataset = val_dataset # using only validation dataset for quantization
    quantized_latents, cluster_indices_hard_assign, cluster_indices_soft_assign = [], [], []
    for batch in out_dataset:
        output, Zq, out_P_proj, _, _ = model(batch, training=False)
        print(f"\nBatch shape: {batch.shape}")
        print(f"\nOutput shape: {output.shape}")
        print(f"\nZq shape: {Zq.shape}")
        print(f"\nout_P_proj shape: {out_P_proj.shape}")

        quantized_latents.append(Zq.numpy())
        # Append soft assignments
        cluster_indices_soft_assign.append(out_P_proj.numpy()) # shows the probability of each index
        cluster_indices_hard_assign.append(tf.argmax(out_P_proj, axis=-1).numpy()) # shows which index has the highest probability
    quantized_latent = np.concatenate(quantized_latents, axis=0)
    cluster_indices_soft = np.concatenate(cluster_indices_soft_assign, axis=0)
    cluster_indices_hard = np.concatenate(cluster_indices_hard_assign, axis=0)
    np.save(os.path.join(output_dir, 'quantized_latent.npy'), quantized_latent)
    # np.save(os.path.join(output_dir, 'cluster_indices_hard.npy'), cluster_indices_hard)

    print(f"\n head of Cluster indices hard: {cluster_indices_hard[:5]}")
    print(f"\n head of Cluster indices soft: {cluster_indices_soft[:5]}")
    # print shape of soft cluster indices
    print(f"\n softs shape: {np.array(cluster_indices_soft).shape}")
    print(f"\n head of Quantized latents: {quantized_latent[:5]}")

    # Create distribution plots for all data
    print("\nCreating distribution plots for all processed data...")
    plot_cluster_distribution(cluster_indices_soft, save_path=os.path.join(output_dir, 'cluster_distribution_soft.png'))
    plot_codebook_usage(cluster_indices_hard, num_embeddings,
                        save_path=os.path.join(output_dir, 'full_codebook_usage.png'))
    plot_soft_cluster_distribution(np.array(cluster_indices_soft), 20, 'soft_cluster_distribution.png')
    print(f"Latent representations saved to {output_dir}")

if __name__ == "__main__":
    # Call train_and_evaluate_scqvae without trying to unpack return values
    train_and_evaluate_scqvae(
        data_path="data/Pep2Vec/ConvNeXT-MHC/pep2vec_output_fold_1.parquet",
        num_embeddings=32,
        embedding_dim=64,
        batch_size=32,
        epochs=20,
        output_dir="data/SCQvae/Conbotnet",
        visualize=True,
        save_model=True,
        random_state=42,
        test_size=0.2,
        output_data="all",  # Options: "val", "train", "all"
    )

# # Create a 1D UMAP projection colored by cluster indices
# print("Computing 1D UMAP colored by cluster indices...")
# # Use the same 1D UMAP projection, but color by cluster indices
# plt.figure(figsize=(12, 4))
# cluster_indices_flat = cluster_indices.flatten().reshape(-1, 1)
# embedding_1d = mapper_1d.fit_transform(cluster_indices_flat)
#
# # Regenerate y_jitter to match the new embedding size
# y_jitter = np.random.rand(embedding_1d.shape[0]) * 0.1
#
# # Check shapes and ensure they match
# print(f"embedding_1d shape: {embedding_1d.shape}")
# print(f"cluster_indices shape: {cluster_indices.shape}")
#
# # Ensure cluster_indices is properly shaped for plotting
# if cluster_indices.ndim > 1:
#     cluster_indices_plot = cluster_indices.flatten()
# else:
#     cluster_indices_plot = cluster_indices
#
# # Make sure lengths match
# if len(cluster_indices_plot) != len(embedding_1d):
#     # Reshape or slice cluster_indices to match embedding_1d
#     if len(cluster_indices_plot) > embedding_1d.shape[0]:
#         cluster_indices_plot = cluster_indices_plot[:embedding_1d.shape[0]]
#     else:
#         # If you need to extend, you might repeat values or use a different strategy
#         cluster_indices_plot = np.pad(cluster_indices_plot, (0, embedding_1d.shape[0] - len(cluster_indices_plot)), 'edge')
#     print(f"Adjusted cluster_indices shape: {cluster_indices_plot.shape}")
#
# scatter_clusters = plt.scatter(embedding_1d[:, 0], y_jitter, c=cluster_indices_plot, cmap='viridis', s=10, alpha=0.7)
# cbar = plt.colorbar(scatter_clusters, label='Cluster Index')
# plt.title('1D UMAP Projection Colored by Cluster Indices', fontsize=14)
# plt.xlabel('UMAP Dimension 1', fontsize=12)
# plt.yticks([])
# plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.savefig('umap_clusters.png')
# plt.show()

# # Apply UMAP dimensionality reduction
# print("Computing UMAP projection...")
# mapper = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42)
# embedding = mapper.fit_transform(latent_2d)
#
# # Visualize the embedding with distinct colors for each sample
# plt.figure(figsize=(12, 10))
#
# # Use sample IDs for coloring (each sample gets a unique color)
# scatter = plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=sample_ids,
#     cmap='tab20',  # Colormap with distinct colors
#     s=10,
#     alpha=0.7
# )
#
# # Add legend and labels
# cbar = plt.colorbar(scatter, label='Sample ID')
# cbar.set_label('Sample ID')
# plt.title('UMAP Projection of Quantized Latent Space (colored by sample)', fontsize=14)
# plt.xlabel('UMAP Dimension 1', fontsize=12)
# plt.ylabel('UMAP Dimension 2', fontsize=12)
#
# # Add a grid for better readability
# plt.grid(linestyle='--', alpha=0.6)
#
# # Save with higher DPI for better quality
# plt.savefig('scqvae_latent_umap_by_sample.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # Create a second visualization colored by cluster assignment
# plt.figure(figsize=(12, 10))
# scatter = plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=cluster_indices,
#     cmap='viridis',
#     s=10,
#     alpha=0.7
# )
#
# # Add legend and labels
# cbar = plt.colorbar(scatter, label='Codebook Vector Index')
# plt.title('UMAP Projection of Quantized Latent Space (colored by codebook vector)', fontsize=14)
# plt.xlabel('UMAP Dimension 1', fontsize=12)
# plt.ylabel('UMAP Dimension 2', fontsize=12)
# plt.grid(linestyle='--', alpha=0.6)
# plt.savefig('scqvae_latent_umap_by_cluster.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # Save the embeddings for potential further analysis
# np.savez(
#     'umap_results.npz',
#     embedding=embedding,
#     sample_ids=sample_ids,
#     cluster_indices=cluster_indices,
#     latent_vectors=latent_2d
# )
# print("UMAP visualization complete. Results saved with sample-based coloring.")
