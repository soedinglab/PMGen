'''import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.model import SCQ_model
import keras

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

# example of running VQUnet
from utils.model import SCQ1DAutoEncoder

# TODO implement a simple training pipeline for VQUnet
import tensorflow as tf
import numpy as np
import time # To time training
import pandas as pd


# --- Configuration ---
# INPUT_SHAPE = (64, 64, 1) # IMPORTANT: Adjust to your MHC1 data's shape (height, width, channels)
NUM_EMBEDDINGS = 16       # Number of clusters/codes in the codebook
EMBEDDING_DIM = 64         # Dimension of each codebook vector (latent dim in bottleneck)
BATCH_SIZE = 4
EPOCHS = 10               # Number of training epochs
LEARNING_RATE = 1e-3

# --- Data Loading and Preparation ---
# def load_mhc1_placeholder_data(num_samples, shape):
#     """Generates random placeholder data."""
#     print(f"Generating {num_samples} placeholder samples with shape {shape}...")
#     # Generate random data (e.g., pixel values between 0 and 1)
#     data = np.random.rand(num_samples, *shape).astype(np.float32)
#     print("Placeholder data generated.")
#     return data

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

data_path = "data/Pep2Vec/wrapper_mhc1.parquet"
# Load or generate data
print("Loading/Generating Training Data...")
train_data, test_data, seq_length, feature_dim = load_data(data_path, test_size=0.2, random_state=42) # Load training data

# Create tf.data Datasets for efficient training
# For reconstruction, input and target are the same (x, x)
print("Creating TensorFlow Datasets...")
train_dataset = create_dataset(train_data, batch_size=BATCH_SIZE, is_training=True)
val_dataset = create_dataset(test_data, batch_size=BATCH_SIZE, is_training=False)
print("Datasets created.")


# --- Model Instantiation ---
print("Building the SCQ1DAutoEncoder model...")
input_shape = (seq_length, feature_dim)

model = SCQ1DAutoEncoder(
    input_dim=input_shape,
    num_embeddings=NUM_EMBEDDINGS,
    embedding_dim=EMBEDDING_DIM,
    commitment_beta=0.25,
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

print("Model built.")

# --- Compile the Model ---
# The loss calculation is handled within the train_step,
# but we still need to provide an optimizer.
print("Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
print("Model compiled.")

# --- Training ---
print(f"Starting training for {EPOCHS} epochs...")
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset # Pass validation data here
)

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- (Optional) Post-Training ---
print("\nTraining History:")
print(history.history)

# Example: Get reconstruction and latent codes for a batch from validation set
print("\nExample inference on validation data:")
example_batch = next(iter(val_dataset))  # Get one batch
output = model.predict(example_batch)
if isinstance(output, (list, tuple)) and len(output) == 3:
    reconstruction, quantized_latent, cluster_indices = output
else:
    reconstruction = output

# Visualize or save the reconstruction
end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

# --- Visualize Training Metrics ---
import matplotlib.pyplot as plt


def plot_training_metrics(history):
        """Plot the training metrics over epochs."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

        # Plot total loss
        axes[0].plot(history.history['total_loss'], 'b-', label='Train')
        if 'val_total_loss' in history.history:
            axes[0].plot(history.history['val_total_loss'], 'r-', label='Validation')
        axes[0].set_title('Total Loss')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()

        # Plot reconstruction loss
        axes[1].plot(history.history['recon_loss'], 'b-', label='Train')
        if 'val_recon_loss' in history.history:
            axes[1].plot(history.history['val_recon_loss'], 'r-', label='Validation')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True)
        axes[1].legend()

        # Plot VQ loss
        axes[2].plot(history.history['vq_loss'], 'b-', label='Train')
        if 'val_vq_loss' in history.history:
            axes[2].plot(history.history['val_vq_loss'], 'r-', label='Validation')
        axes[2].set_title('VQ Loss')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)
        axes[2].legend()

        # Plot perplexity
        axes[3].plot(history.history['perplexity'], 'b-', label='Train')
        if 'val_perplexity' in history.history:
            axes[3].plot(history.history['val_perplexity'], 'r-', label='Validation')
        axes[3].set_title('Perplexity')
        axes[3].set_xlabel('Epochs')
        axes[3].set_ylabel('Perplexity')
        axes[3].grid(True)
        axes[3].legend()

        plt.tight_layout()
        plt.savefig('vqvae_training_metrics.png')
        plt.show()


# Plot the training metrics
print("\nPlotting training history...")
plot_training_metrics(history)

# --- Example Inference and Visualization ---
print("\nPerforming example inference on validation data...")
example_batch = next(iter(val_dataset))  # Get one batch
output = model(example_batch, training=False)

# Unpack the output
reconstruction, quantized_latent, cluster_indices, vq_loss, perplexity= output


# Visualize reconstructions
def plot_reconstructions(original, reconstructed, n_samples=5):
    """Plot comparison between original and reconstructed sequences."""
    n_samples = min(n_samples, len(original))
    plt.figure(figsize=(15, 3 * n_samples))

    for i in range(n_samples):
        # Plot original sequence
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.plot(original[i, :, 0])  # Assuming last dim is feature dim with size 1
        plt.title(f"Original Sequence {i + 1}")
        plt.grid(True)

        # Plot reconstructed sequence
        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.plot(reconstructed[i, :, 0])  # Assuming same shape as original
        plt.title(f"Reconstructed Sequence {i + 1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('vqvae_reconstructions.png')
    plt.show()


print("\nVisualizing reconstructions...")
plot_reconstructions(example_batch.numpy(), reconstruction.numpy())


# Visualize codebook usage distribution
def plot_codebook_usage(indices, num_embeddings, save_path='codebook_usage.png'):
        """
        Visualize the usage distribution of codebook vectors.
        # we have a list of float values around 100 unique values, but we only have 32 codebook vectors
        # we need to assign the float values to the codebook vectors with a deviation threshold based on num_embeddings
        # define N ranges between 0 to 1 such that N == num_embeddings
        # assign the float values to the ranges and replace the float values with the index of the range

        Args:
            indices: Tensor containing the indices of the codebook vectors used
            num_embeddings: Total number of vectors in the codebook
            save_path: Path to save the visualization
        """

        # Convert to numpy if it's a tensor
        if isinstance(indices, tf.Tensor):
            indices = indices.numpy()

        # Flatten the indices if they're not already flattened
        flat_indices = indices.flatten()

        # Check if we need to quantize float values to discrete indices
        if np.issubdtype(flat_indices.dtype, np.floating):
            print(f"Detected floating point indices, quantizing to {num_embeddings} discrete values")
            min_val = np.min(flat_indices)
            max_val = np.max(flat_indices)
            print(f"Index range: {min_val} to {max_val}")

            # Create quantization bins
            bins = np.linspace(min_val, max_val, num_embeddings + 1)

            # Digitize the indices (assign each to a bin)
            discrete_indices = np.digitize(flat_indices, bins) - 1

            # Clip to ensure valid range
            discrete_indices = np.clip(discrete_indices, 0, num_embeddings - 1)
            flat_indices = discrete_indices

        # Count occurrences of each codebook vector
        unique, counts = np.unique(flat_indices, return_counts=True)

        # Create a complete distribution including zeros for unused vectors
        full_distribution = np.zeros(num_embeddings)
        for idx, count in zip(unique, counts):
            if 0 <= idx < num_embeddings:
                full_distribution[int(idx)] = count

        # Calculate usage statistics
        used_vectors = np.sum(full_distribution > 0)
        usage_percentage = (used_vectors / num_embeddings) * 100

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Create bar plot
        bar_positions = np.arange(num_embeddings)
        bars = plt.bar(bar_positions, full_distribution)

        # Add a color gradient based on frequency
        max_count = np.max(full_distribution)
        if max_count > 0:  # Avoid division by zero
            for i, bar in enumerate(bars):
                intensity = full_distribution[i] / max_count
                bar.set_color(plt.cm.viridis(intensity))

        # Add labels and title
        plt.xlabel('Codebook Vector Index')
        plt.ylabel('Usage Count')
        plt.title(f'Codebook Vector Usage Distribution\n{used_vectors}/{num_embeddings} vectors used ({usage_percentage:.1f}%)')

        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add a colorbar as a usage intensity reference
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, max_count))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Frequency')

        # Improve layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)

        # Display statistics
        print(f"Codebook usage statistics:")
        print(f"- Total vectors in codebook: {num_embeddings}")
        print(f"- Number of vectors used: {used_vectors} ({usage_percentage:.1f}%)")
        if used_vectors > 0:
            print(f"- Most used vector: {np.argmax(full_distribution)} (used {np.max(full_distribution)} times)")
            used_indices = np.where(full_distribution > 0)[0]
            min_used_idx = used_indices[np.argmin(full_distribution[used_indices])]
            print(f"- Least used vector: {min_used_idx} (used {full_distribution[min_used_idx]} times)")

        plt.show()

        return used_vectors, usage_percentage



print("\nAnalyzing codebook usage...")
plot_codebook_usage(cluster_indices, NUM_EMBEDDINGS)

# Save the model
model.save_weights('tmp_directory/vqvae_model_weights.h5')
print("\nModel weights saved to 'vqvae_model_weights.h5'")

# You can now save the model weights if needed
# model.save_weights('vq_unet_mhc1_weights.h5')
# print("Model weights saved.")

# TODO draw a U-MAP plot of the latent space
