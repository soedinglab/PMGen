#!/usr/bin/env python
"""
=========================

End‑to‑end trainer for a **peptide×MHC cross‑attention classifier**.
It loads a NetMHCpan‑style parquet that contains

    long_mer, assigned_label, allele, MHC_class,
    mhc_embedding  **OR**  mhc_embedding_path

columns.  Each row supplies

* a peptide sequence (long_mer)
* a pre‑computed MHC pseudo‑sequence embedding (36, 1152)
* a binary label (assigned_label)

The script

1.Derives the longest peptide length → SEQ_LEN.
2.Converts every peptide into a 21‑dim one‑hot tensor (SEQ_LEN, 21).
3.Feeds the pair

        (one_hot_peptide, mhc_latent) → classifier → P(binding)

4.Trains with binary‑cross‑entropy and saves the best weights & metadata.

Author :  Amirreza (updated for cross‑attention, 2025‑05‑22)
"""
from __future__ import annotations
import os
import sys

print(sys.executable)

# =============================================================================
# CRITICAL: GPU Memory Configuration - MUST BE FIRST
# =============================================================================
import tensorflow as tf

def configure_gpu_memory():
    """Configure TensorFlow to use GPU memory efficiently"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        else:
            print("No GPUs found - running on CPU")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


# Configure GPU immediately
configure_gpu_memory()

# Set memory-friendly environment variables
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import math
import argparse, datetime, pathlib, json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_archive import build_custom_classifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)
import seaborn as sns
import os
import pyarrow.parquet as pq

# =============================================================================
# Memory monitoring functions
# =============================================================================
def monitor_memory():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB ({memory.percent:.1f}% used)")

    try:
        # Try to get GPU memory info
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(
                f"GPU {i}: {info.used / 1e9:.1f}GB / {info.total / 1e9:.1f}GB ({100 * info.used / info.total:.1f}% used)")
    except:
        print("GPU memory monitoring not available")


def check_memory():
    """Check if memory usage is acceptable"""
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        print(f"⚠️  Critical RAM usage: {memory.percent:.1f}%")
        return False
    elif memory.percent > 80:
        print(f"⚠️  High RAM usage: {memory.percent:.1f}%")
    return True

# ----------------------------------------------------------------------------
# 1) Peptide → k‑mer one‑hot *inside* TF graph (GPU/TPU friendly)
# ----------------------------------------------------------------------------
pad_token = 20
AA = tf.constant(list("ACDEFGHIKLMNPQRSTVWY"))
TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(AA, tf.range(20, dtype=tf.int32)),
    default_value=tf.constant(pad_token, dtype=tf.int32),  # UNK / padding
)

def kmer_onehot_tensor(seqs: tf.Tensor, seq_len: int, k: int = 9) -> tf.Tensor:
    """Vectorised k‑mer framing + one‑hot with no Python loops."""
    tokens  = tf.strings.bytes_split(seqs)               # ragged ‹N,L›
    idx     = TABLE.lookup(tokens.flat_values)
    ragged  = tf.RaggedTensor.from_row_lengths(idx, tokens.row_lengths())
    idx_pad = ragged.to_tensor(default_value=pad_token, shape=(None, seq_len))  # (N,L)
    frames  = tf.signal.frame(idx_pad, frame_length=k, frame_step=1, axis=1)  # (N,RF,k)
    return tf.one_hot(frames, depth=21, dtype=tf.float32)                 # (N,RF,k,21)

# ---------------------------------------------------------------------------
# tf.data convenience wrapper ----------------------------------------------
# ---------------------------------------------------------------------------

def _parquet_rowcount(parquet_path: str | os.PathLike) -> int:
    return pq.ParquetFile(parquet_path).metadata.num_rows

# ---------------------------------------------------------------------------
# Robust loader for latent embeddings (same as before)
# ---------------------------------------------------------------------------

def _read_embedding_file(path: str | os.PathLike) -> np.ndarray:
    # Try fast numeric path first
    try:
        arr = np.load(path)
        if isinstance(arr, np.ndarray) and arr.dtype == np.float32:
            return arr
        raise ValueError
    except ValueError:
        obj = np.load(path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()
        if isinstance(obj, dict) and "embedding" in obj:
            return obj["embedding"].astype("float32")
        raise ValueError(f"Unrecognised embedding file {path}")


# ----------------------------------------------------------------------------
# 3) Streaming Parquet → tf.data.Dataset (stays in TF tensors the whole time)
# ----------------------------------------------------------------------------


def _load_parquet_samples(parquet_path, seq_len, k=9):
    """
    FIXED: Generator that yields individual samples, not batches.
    This prevents double-batching issues.
    """
    pq_file = pq.ParquetFile(parquet_path)
    total_rows = _parquet_rowcount(parquet_path)
    estimated_batches = math.ceil(total_rows / BUFFER)
    for batch_df in tqdm(pq_file.iter_batches(batch_size=BUFFER),
                         total=estimated_batches,
                         desc="Loading batches",
                         unit="batch"):
        df = batch_df.to_pandas()
        # Process each row individually
        # Vectorized batch processing
        seqs = tf.constant(df["long_mer"].tolist(), dtype=tf.string) # (N,)
        labels = tf.constant(df["assigned_label"].values, dtype=tf.float32) # (N,)
        # Load and stack all embeddings for the batch
        embeddings = [_read_embedding_file(p) for p in df["mhc_embedding_path"]]
        latents = tf.constant(np.stack(embeddings), dtype=tf.float32) # (N, 1152)
        # One-hot encode all peptides in one go
        x_peps = kmer_onehot_tensor(seqs, seq_len=seq_len, k=k) # (N, RF, k, 21)
        # Yield individual samples
        for x_pep, latent, label in zip(x_peps, latents, labels):
            yield (x_pep, latent), tf.expand_dims(label, 0)
        # Clean up DataFrame
        del df

def make_stream_dataset(
        parquet_path: str,
        max_pep_seq_len: int,
        max_mhc_len: int,
        k: int = 9,
) -> tf.data.Dataset:
    """
    FIXED: Memory-optimized dataset creation with proper batching.
    """
    rf = max_pep_seq_len - k + 1

    # Output signature for individual samples (no batch dimension)
    output_signature = (
        (
            tf.TensorSpec(shape=(rf, k, 21), dtype=tf.float32),  # No batch dim
            tf.TensorSpec(shape=(max_mhc_len, 1152), dtype=tf.float32),  # No batch dim
        ),
        tf.TensorSpec(shape=1, dtype=tf.float32),  # Scalar label
    )
    ds = tf.data.Dataset.from_generator(
        lambda: _load_parquet_samples(parquet_path, max_pep_seq_len, k),
        output_signature=output_signature,
    )
    return ds

# ---------------------------------------------------------------------------
# Visualisation utility
# ---------------------------------------------------------------------------

def plot_training_curve(history: tf.keras.callbacks.History, run_dir: str, fold_id: int = None,
                        model=None, val_dataset=None):
    """
    Plot training curves and validation metrics from a Keras history object.

    Args:
        history: Keras history object containing training metrics.
        run_dir: Directory to save the plot.
        fold_id: Optional fold identifier for naming the output file.
        model: Optional model to compute confusion matrix and ROC curve.
        val_dataset: Optional validation dataset to generate predictions.

    Returns: None
    """
    hist = history.history
    plt.figure(figsize=(21, 6))
    plot_name = f"training_curve{'_fold' + str(fold_id) if fold_id is not None else ''}"

    # set plot name above all plots
    plt.suptitle(f"Training Curves{' (Fold ' + str(fold_id) + ')' if fold_id is not None else ''}", fontsize=16, fontweight='bold')

    # Plot 1: Loss curve
    plt.subplot(1, 4, 1)
    plt.plot(hist["loss"], label="train", linewidth=2)
    plt.plot(hist["val_loss"], label="val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Precision - Recall curve
    # if "Precision" in hist and "Recall" in hist:
    #     plt.subplot(1, 4, 2)
    #     plt.plot(hist["Recall"], hist["Precision"], label="PR Curve", linewidth=2)
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.title("Precision-Recall Curve")
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)

    # Plot 2: Accuracy curve
    if "binary_accuracy" in hist and "val_binary_accuracy" in hist:
        plt.subplot(1, 4, 2)
        plt.plot(hist["binary_accuracy"], label="train acc", linewidth=2)
        plt.plot(hist["val_binary_accuracy"], label="val acc", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Binary Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
    elif "accuracy" in hist and "val_accuracy" in hist:
        plt.subplot(1, 4, 2)
        plt.plot(hist["accuracy"], label="train acc", linewidth=2)
        plt.plot(hist["val_accuracy"], label="val acc", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: AUC curve
    if "AUC" in hist and "val_AUC" in hist:
        plt.subplot(1, 4, 3)
        plt.plot(hist["AUC"], label="train AUC", linewidth=2)
        plt.plot(hist["val_AUC"], label="val AUC", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("AUC")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 4: Confusion matrix (if model and validation dataset provided)
    if model is not None and val_dataset is not None:
        plt.subplot(1, 4, 4)

        # Get predictions
        y_pred_proba = model.predict(val_dataset)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Extract true labels
        y_true = []
        for _, labels in val_dataset:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        # print ranges of y_true and y_pred
        print(f"y_true range: {y_true.min()} to {y_true.max()}")

        print(f"y_pred range: {y_pred.min()} to {y_pred.max()}")

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    else:
        # If no model/dataset provided, show a placeholder or additional metric
        plt.subplot(1, 4, 4)
        plt.text(0.5, 0.5, 'Confusion Matrix\n(Requires model + val_dataset)',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.axis('off')

    # Save main plot
    plt.tight_layout()
    out_png = os.path.join(run_dir, f"{plot_name}.png")

    # Create directory if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Training curve saved to {out_png}")


def plot_test_metrics(model, test_dataset, run_dir: str, fold_id: int = None, history=None, string: str =None):
    """
    Plot comprehensive evaluation metrics for a test dataset using a trained model.

    Args:
        model: Trained Keras model.
        test_dataset: TensorFlow dataset for testing.
        run_dir: Directory to save the plot.
        fold_id: Optional fold identifier for naming the output file.
        history: Optional training history object to display loss/accuracy curves.

    Returns: Dictionary containing evaluation metrics
    """
    # Collect predictions
    print("Generating predictions...")
    y_pred_proba = model.predict(test_dataset)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Extract true labels
    y_true = []
    for _, labels in test_dataset:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true).flatten()

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba.flatten())
    roc_auc = auc(fpr, tpr)

    # Create a multi-panel figure
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{string} Evaluation Metrics{' (Fold ' + str(fold_id) + ')' if fold_id is not None else ''}",
                 fontsize=16, fontweight='bold')

    # Panel 1: ROC Curve
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Panel 2: Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Panel 3: Loss curve (if history provided)
    if history is not None:
        plt.subplot(2, 2, 3)
        hist = history.history
        plt.plot(hist.get("loss", []), label="train", linewidth=2)
        plt.plot(hist.get("val_loss", []), label="val", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("BCE Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Panel 4: Accuracy curve (if available in history)
        plt.subplot(2, 2, 4)
        if "binary_accuracy" in hist and "val_binary_accuracy" in hist:
            plt.plot(hist["binary_accuracy"], label="train acc", linewidth=2)
            plt.plot(hist["val_binary_accuracy"], label="val acc", linewidth=2)
        elif "accuracy" in hist and "val_accuracy" in hist:
            plt.plot(hist["accuracy"], label="train acc", linewidth=2)
            plt.plot(hist["val_accuracy"], label="val acc", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Panel 3: Test metrics when history not available
        plt.subplot(2, 2, 3)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': roc_auc
        }

        # Create a bar chart for metrics
        bars = plt.bar(range(len(metrics)), list(metrics.values()),
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'],
                       alpha=0.8, edgecolor='black', linewidth=1)
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.title('Test Set Evaluation Metrics')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3, axis='y')

        # Add values on top of bars
        for i, (bar, value) in enumerate(zip(bars, metrics.values())):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # Panel 4: Prediction distribution
        plt.subplot(2, 2, 4)
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7,
                 label='Negative Class', color='red', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7,
                 label='Positive Class', color='blue', density=True)
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                    label='Decision Threshold')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)

    out_png = os.path.join(run_dir, f"{string}_metrics{'_fold' + str(fold_id) if fold_id is not None else ''}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Test metrics visualization saved to {out_png}")

    # Calculate and return evaluation metrics
    metrics_dict = {
        'roc_auc': roc_auc,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall:    {metrics_dict['recall']:.4f}")
    print(f"F1 Score:  {metrics_dict['f1']:.4f}")
    print(f"ROC AUC:   {metrics_dict['roc_auc']:.4f}")
    print("=" * 50)

    return metrics_dict

# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=False,
                   help="Input parquet with peptide, latent, label")
    p.add_argument("--dataset_path", default=None,
                   help="Path to a custom dataset (not parquet)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--outdir", default=None,
                   help="Output dir (default: runs/run_YYYYmmdd-HHMMSS)")
    p.add_argument("--test_batches", type=int, default=None,
                   help="Limit to N batches for testing (default: None = use all data)")

    args = p.parse_args(argv)

    run_dir = args.outdir or f"runs/run_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"★ Outputs → {run_dir}\n")

    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    print("Setting random seeds for reproducibility...")

    # ----------------------- Load & split ----------------------------------
    # Find the longest peptide sequence across all datasets
    longest_peptide_seq_length = 0
    max_mhc_seq_length = 36  # TODO make dynamic later

    # Load test sets
    test1 = pd.read_parquet(args.dataset_path + "/test1.parquet")
    test2 = pd.read_parquet(args.dataset_path + "/test2.parquet")
    esm_files = list(pathlib.Path(args.dataset_path).glob('*esm_embeddings.parquet'))
    if not esm_files:
        raise FileNotFoundError('No esm embeddings parquet found in dataset_path')
    whole_data = pd.read_parquet(str(esm_files[0]))

    fold_files = sorted([f for f in os.listdir(os.path.join(args.dataset_path, f'folds')) if f.endswith('.parquet')])
    n_folds = len(fold_files) // 2

    # Check test datasets
    if "long_mer" in test1.columns:
        longest_peptide_seq_length = max(longest_peptide_seq_length, int(test1["long_mer"].str.len().max()))

    if "long_mer" in test2.columns:
        longest_peptide_seq_length = max(longest_peptide_seq_length, int(test2["long_mer"].str.len().max()))
    # Check whole dataset
    if "long_mer" in whole_data.columns:
        longest_peptide_seq_length = max(longest_peptide_seq_length, int(whole_data["long_mer"].str.len().max()))

    # Create fold datasets with consistent sequence length
    folds = []
    class_weights = []
    for i in range(1, n_folds + 1):
        print(f"processing fold {i}")
        train_path = os.path.join(args.dataset_path, f'folds/fold_{i}_train.parquet')
        val_path = os.path.join(args.dataset_path, f'folds/fold_{i}_val.parquet')

        train_ds = (make_stream_dataset(train_path, max_pep_seq_len=longest_peptide_seq_length,
                                        max_mhc_len=max_mhc_seq_length)
                    .shuffle(buffer_size=BUFFER, reshuffle_each_iteration=True)
                    .batch(args.batch)
                    .take(args.test_batches)
                    .prefetch(tf.data.AUTOTUNE))

        val_ds = (make_stream_dataset(val_path, max_pep_seq_len=longest_peptide_seq_length,
                                      max_mhc_len=max_mhc_seq_length)
                  .batch(args.batch)
                  .take(args.test_batches)
                  .prefetch(tf.data.AUTOTUNE))

        # Calculate class weights (unchanged)
        train_labels = pd.read_parquet(train_path)["assigned_label"].values.astype(int)
        counts = np.bincount(train_labels, minlength=2)
        cw = {0: 1.0, 1: 1.0}
        if counts[0] > 0 and counts[1] > 0:
            total = counts.sum()
            cw[0] = total / (2 * counts[0])
            cw[1] = total / (2 * counts[1])

        folds.append((train_ds, val_ds))
        class_weights.append(cw)

    # Test loaders
    test1_ds = (make_stream_dataset(os.path.join(args.dataset_path, "test1.parquet"),
                                    longest_peptide_seq_length, max_mhc_seq_length)
                .batch(args.batch)
                .prefetch(tf.data.AUTOTUNE))

    test2_ds = (make_stream_dataset(os.path.join(args.dataset_path, "test2.parquet"),
                                    longest_peptide_seq_length, max_mhc_seq_length)
                .batch(args.batch)
                .prefetch(tf.data.AUTOTUNE))

    print(f"✓ loaded {len(test1):,} test1 samples, "
          f"{len(test2):,} test2 samples, "
          f"{len(folds)} folds")

    # ----------------------- Model -----------------------------------------
    # clear GPU memory
    tf.keras.backend.clear_session()
    print("Memory before fold processing:")
    monitor_memory()
    # ------------------------- TRAIN --------------------------------------
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(run_dir, 'best.weights.h5'),
        monitor='val_loss', save_best_only=True, mode='min')
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    for fold_id, ((train_loader, val_loader), class_weight) in enumerate(zip(folds, class_weights), start=1):
        print(f'Training on fold {fold_id}/{len(folds)}')
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        print("########################### seq length: ", longest_peptide_seq_length)
        print(f"redefining the model in each fold")
        model = build_custom_classifier(longest_peptide_seq_length, max_mhc_seq_length, pad_token=pad_token)
        model.summary()

        # print shapes
        for (x_pep, latents), labels in train_loader.take(1):
            print(f"Input shapes: x_pep={x_pep.shape}, latents={latents.shape}, labels={labels.shape}")

        print("Training model...")
        hist = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs= args.epochs,
            class_weight=class_weight,
            callbacks=[ckpt_cb, early_cb],
            verbose=1,
        )


        # Gradient Tape method
        # optimizer = tf.keras.optimizers.Adam()
        # loss_fn = tf.keras.losses.BinaryCrossentropy(
        #     from_logits=False)  # Use from_logits=False if your model has a sigmoid output
        #
        # best_val_loss = float('inf')
        # patience_counter = 0
        # best_weights = None
        #
        # for epoch in range(args.epochs):
        #     print(f"Epoch {epoch + 1}/{args.epochs}")
        #
        #     # Training loop
        #     train_loss = 0.0
        #     train_batches = 0
        #     for (x_pep, latents), labels in train_loader:
        #         with tf.GradientTape() as tape:
        #             predictions = model([x_pep, latents], training=True)
        #             # Apply class weights manually
        #             sample_weights = tf.where(tf.equal(labels, 1),
        #                                       class_weight[1],
        #                                       class_weight[0])
        #             loss_value = loss_fn(labels, predictions, sample_weight=sample_weights)
        #
        #         gradients = tape.gradient(loss_value, model.trainable_variables)
        #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #
        #         train_loss += loss_value
        #         train_batches += 1
        #
        #     avg_train_loss = train_loss / train_batches
        #
        #     # Validation loop
        #     val_loss = 0.0
        #     val_batches = 0
        #     for (x_pep, latents), labels in val_loader:
        #         predictions = model([x_pep, latents], training=False)
        #         val_loss += loss_fn(labels, predictions)
        #         val_batches += 1
        #
        #     avg_val_loss = val_loss / val_batches
        #
        #     # Print metrics
        #     print(f"Training loss: {avg_train_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
        #
        #     # Implement early stopping and model checkpoint
        #     if avg_val_loss < best_val_loss:
        #         best_val_loss = avg_val_loss
        #         best_weights = model.get_weights()
        #         patience_counter = 0
        #         # Save checkpoint
        #         model.save_weights(os.path.join(run_dir, 'best.weights.h5'))
        #         print(f"✓ Saved best weights (val_loss: {best_val_loss:.4f})")
        #     else:
        #         patience_counter += 1
        #         if patience_counter >= 10:  # Match early_cb patience
        #             print(f"Early stopping triggered after {epoch + 1} epochs")
        #             break
        #
        # # Restore best weights
        # if best_weights is not None:
        #     model.set_weights(best_weights)
        #
        # # Create history object with metrics for compatibility with other code
        # hist = type('obj', (object,), {
        #     'history': {
        #         'loss': [0],  # Placeholder
        #         'val_loss': [best_val_loss]
        #     }
        # })
        ######################

        # plot
        plot_training_curve(hist, run_dir, fold_id, model, val_loader)

        # save model and metadata
        model.save(os.path.join(run_dir, f'model_fold_{fold_id}.weights.h5'))
        metadata = {
            "fold_id": fold_id,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "seq_len": longest_peptide_seq_length,
            "run_dir": run_dir
        }
        with open(os.path.join(run_dir, f'metadata_fold_{fold_id}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Fold {fold_id} model saved to {run_dir}")

        # Evaluate on test sets
        print("Evaluating on test1 set...")

        for (x_pep, latents), labels in test1_ds.take(1):
            print(f"Test1 Input shapes: x_pep={x_pep.shape}, latents={latents.shape}, labels={labels.shape}")

        test1_results = model.evaluate(test1_ds, verbose=1)
        print(f"Test1 results: {test1_results}")
        print("Evaluating on test2 set...")

        for (x_pep, latents), labels in test2_ds.take(1):
            print(x_pep.shape, latents.shape, labels.shape)

        test2_results = model.evaluate(test2_ds, verbose=1)
        print(f"Test2 results: {test2_results}")

        # Plot ROC curve for test1
        plot_test_metrics(model, test1_ds, run_dir, fold_id, string="Test1 - balanced alleles")
        # Plot ROC curve for test2
        plot_test_metrics(model, test2_ds, run_dir, fold_id, string="Test2 - rare alleles")

        # Aggressive cleanup
        del model, hist, train_loader, val_loader


if __name__ == "__main__":
    BUFFER = 512
    main([
        "--dataset_path", "../data/Custom_dataset/NetMHCpan_dataset/mhc_1",
        "--epochs", "3", "--batch", "128", "--test_batches", "5"
    ])