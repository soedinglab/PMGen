import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.model import SCQ_model

# Set TensorFlow logging level for more information
tf.get_logger().setLevel('INFO')


def create_dataset(X, batch_size=1, is_training=True):
    """Create TensorFlow dataset with consistent approach for both training and validation."""
    # Debug info
    print(f"Creating {'training' if is_training else 'validation'} dataset")
    print(f"Data shape: {X.shape}")
    print(f"Data dtype: {X.dtype}")
    print(f"Data range: min={np.min(X):.4f}, max={np.max(X):.4f}, mean={np.mean(X):.4f}")

    # Ensure data is float32 (TensorFlow works best with float32)
    X = X.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(X)

    # Apply shuffling only for training data
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    # Apply batching with drop_remainder=False to handle all data
    dataset = dataset.batch(batch_size)

    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Debug dataset
    for batch in dataset.take(1):
        print(f"Sample batch shape: {batch.shape}")
        print(f"Sample batch dtype: {batch.dtype}")
        print(f"Sample batch range: min={tf.reduce_min(batch):.4f}, max={tf.reduce_max(batch):.4f}")

    return dataset

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
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 8, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 32, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 64, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 128, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 256, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 512, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 1024, 'heads': 4},
        # Explore codebook_dim
        {'general_embed_dim': 128, 'codebook_dim': 32, 'codebook_num': 16, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 64, 'codebook_num': 16, 'heads': 4},
        {'general_embed_dim': 128, 'codebook_dim': 128, 'codebook_num': 16, 'heads': 4},
        # Explore heads
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 2},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 8},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 16},
        {'general_embed_dim': 128, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 32},
        # Explore general_embed_dim
        {'general_embed_dim': 64, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
        {'general_embed_dim': 256, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
        {'general_embed_dim': 512, 'codebook_dim': 16, 'codebook_num': 16, 'heads': 4},
    ]

    # Common parameters for all configurations
    common_params = {
        'descrete_loss': False,
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback

        traceback.print_exc()