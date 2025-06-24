#!/usr/bin/env python
"""
=========================

MEMORY-OPTIMIZED End‑to‑end trainer for a **peptide×MHC cross‑attention classifier**.
Loads NetMHCpan‑style parquet files in true streaming fashion without loading entire datasets into memory.

Key improvements:
1. Streaming parquet reading with configurable batch sizes
2. Lazy evaluation of dataset properties (seq length, class balance)
3. Memory-efficient TensorFlow data pipelines
4. Proper cleanup and memory monitoring

Author: Amirreza (memory-optimized version, 2025)
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

# ---------------------------------------------------------------------
# ► Use all logical CPU cores for TF ops that still run on CPU
# ---------------------------------------------------------------------
NUM_CPUS = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(NUM_CPUS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CPUS)
print(f'✓ TF intra/inter-op threads set to {NUM_CPUS}')

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
from model3 import build_classifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)
import seaborn as sns
import pyarrow.parquet as pq
import gc
import weakref
import pyarrow as pa, pyarrow.compute as pc
pa.set_cpu_count(os.cpu_count())


# =============================================================================
# Memory monitoring functions
# =============================================================================
def monitor_memory():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB ({memory.percent:.1f}% used)")

    try:
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


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass


# ----------------------------------------------------------------------------
# Peptide encoding utilities
# ----------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard AAs, order fixed
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
UNK_IDX = 20  # index for unknown / padding


def peptides_to_onehot(sequence: str, max_seq_len: int) -> np.ndarray:
    """Convert peptide sequence to one-hot encoding"""
    arr = np.zeros((max_seq_len, 21), dtype=np.float32)
    for j, aa in enumerate(sequence.upper()[:max_seq_len]):
        arr[j, AA_TO_IDX.get(aa, UNK_IDX)] = 1.0
    return arr


def _read_embedding_file(path: str | os.PathLike) -> np.ndarray:
    """Robust loader for latent embeddings"""
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
# Streaming dataset utilities
# ----------------------------------------------------------------------------
class StreamingParquetReader:
    """Memory-efficient streaming parquet reader"""

    def __init__(self, parquet_path: str, batch_size: int = 1000):
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self._file = None
        self._num_rows = None

    def __enter__(self):
        self._file = pq.ParquetFile(self.parquet_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file = None

    @property
    def num_rows(self):
        """Get total number of rows without loading data"""
        if self._num_rows is None:
            if self._file is None:
                with pq.ParquetFile(self.parquet_path) as f:
                    self._num_rows = f.metadata.num_rows
            else:
                self._num_rows = self._file.metadata.num_rows
        return self._num_rows

    def iter_batches(self):
        """Iterate over parquet file in batches"""
        if self._file is None:
            raise RuntimeError("Reader not opened. Use within 'with' statement.")

        for batch in self._file.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            yield df
            del df, batch  # Explicit cleanup

    def sample_for_metadata(self, n_samples: int = 1000):
        """Sample a small portion for metadata extraction"""
        with pq.ParquetFile(self.parquet_path) as f:
            # Read first batch for metadata
            first_batch = next(f.iter_batches(batch_size=min(n_samples, self.num_rows)))
            return first_batch.to_pandas()


def get_dataset_metadata(parquet_path: str):
    """Extract dataset metadata without loading full dataset"""
    with StreamingParquetReader(parquet_path) as reader:
        sample_df = reader.sample_for_metadata(reader.num_rows)

        metadata = {
            'total_rows': reader.num_rows,
            'max_peptide_length': int(sample_df['long_mer'].str.len().max()) if 'long_mer' in sample_df.columns else 0,
            'class_distribution': sample_df[
                'assigned_label'].value_counts().to_dict() if 'assigned_label' in sample_df.columns else {},
        }

        del sample_df
        return metadata


def calculate_class_weights(parquet_path: str):
    """Calculate class weights from a sample of the dataset"""
    with StreamingParquetReader(parquet_path, batch_size=1000) as reader:
        label_counts = {0: 0, 1: 0}
        for batch_df in reader.iter_batches():
            batch_labels = batch_df['assigned_label'].values
            unique, counts = np.unique(batch_labels, return_counts=True)
            for label, count in zip(unique, counts):
                if label in [0, 1]:
                    label_counts[int(label)] += count
            del batch_df

    # Calculate balanced class weights
    total = sum(label_counts.values())
    if total == 0 or label_counts[0] == 0 or label_counts[1] == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: total / (2 * label_counts[0]),
        1: total / (2 * label_counts[1])
    }


# ---------------------------------------------------------------------
# Utility that is executed in worker processes
# (must be top-level so it can be pickled on Windows)
# ---------------------------------------------------------------------
def _row_to_tensor_pack(row_dict: dict, max_pep_seq_len: int, max_mhc_len: int):
    """Convert a single row (already in plain-python dict form) into tensors."""
    # --- peptide one-hot ------------------------------------------------
    pep = row_dict["long_mer"].upper()[:max_pep_seq_len]
    pep_arr = np.zeros((max_pep_seq_len, 21), dtype=np.float32)
    for j, aa in enumerate(pep):
        pep_arr[j, AA_TO_IDX.get(aa, UNK_IDX)] = 1.0

    # --- load MHC embedding --------------------------------------------
    mhc = _read_embedding_file(row_dict["mhc_embedding_path"])
    if mhc.shape[0] != max_mhc_len:               # sanity check
        raise ValueError(f"MHC length mismatch: {mhc.shape[0]} vs {max_mhc_len}")

    # --- label ----------------------------------------------------------
    label = float(row_dict["assigned_label"])
    return (pep_arr, mhc.astype("float32")), label

from concurrent.futures import ProcessPoolExecutor
import functools, itertools

def streaming_data_generator(
        parquet_path: str,
        max_pep_seq_len: int,
        max_mhc_len: int,
        batch_size: int = 1000):
    """
    Yields *individual* samples, but converts an entire Parquet batch
    on multiple CPU cores first.
    """
    with StreamingParquetReader(parquet_path, batch_size) as reader, \
         ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:

        # Partial function to avoid re-sending constants
        worker_fn = functools.partial(
            _row_to_tensor_pack,
            max_pep_seq_len=max_pep_seq_len,
            max_mhc_len=max_mhc_len,
        )

        for batch_df in reader.iter_batches():
            # Convert Arrow table → list[dict] once; avoids pandas overhead
            dict_rows = batch_df.to_dict(orient="list")      # columns -> python lists
            # Re-shape to list[dict(row)]
            rows_iter = ( {k: dict_rows[k][i] for k in dict_rows}  # row dict
                          for i in range(len(batch_df)) )

            # Parallel map; chunksize tuned for large batches
            results = pool.map(worker_fn, rows_iter, chunksize=64)

            # Stream each converted sample back to the generator consumer
            yield from results      # <-- keeps memory footprint tiny

            # explicit clean-up
            del batch_df, dict_rows, rows_iter, results


def create_streaming_dataset(parquet_path: str,
                             max_pep_seq_len: int,
                             max_mhc_len: int,
                             batch_size: int = 128,
                             buffer_size: int = 1000):
    """
    Same semantics as before, but the generator already does parallel
    preprocessing.  We now ask tf.data to interleave multiple generator
    shards in parallel as well.
    """
    output_signature = (
        (
            tf.TensorSpec(shape=(max_pep_seq_len, 21),  dtype=tf.float32),
            tf.TensorSpec(shape=(max_mhc_len, 1152),    dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: streaming_data_generator(
            parquet_path,
            max_pep_seq_len,
            max_mhc_len,
            buffer_size),
        output_signature=output_signature,
    )

    # ► Parallel interleave gives another speed-up if the Parquet file has
    #   many row-groups – adjust cycle_length as needed.
    ds = ds.interleave(
        lambda x, y: tf.data.Dataset.from_tensors((x, y)),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    return ds


# ----------------------------------------------------------------------------
# Visualization utilities (keeping the same as original)
# ----------------------------------------------------------------------------
def plot_training_curve(history: tf.keras.callbacks.History, run_dir: str, fold_id: int = None,
                        model=None, val_dataset=None):
    """Plot training curves and validation metrics"""
    hist = history.history
    plt.figure(figsize=(21, 6))
    plot_name = f"training_curve{'_fold' + str(fold_id) if fold_id is not None else ''}"

    plt.suptitle(f"Training Curves{' (Fold ' + str(fold_id) + ')' if fold_id is not None else ''}",
                 fontsize=16, fontweight='bold')

    # Plot 1: Loss curve
    plt.subplot(1, 4, 1)
    plt.plot(hist["loss"], label="train", linewidth=2)
    plt.plot(hist["val_loss"], label="val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

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

    # Plot 4: Confusion matrix placeholder
    plt.subplot(1, 4, 4)
    if model is not None and val_dataset is not None:
        # Sample a subset for confusion matrix to avoid memory issues
        sample_dataset = val_dataset.take(100)  # Take only 100 batches
        y_pred_proba = model.predict(sample_dataset, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)

        y_true = []
        for _, labels in sample_dataset:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix (100 Batches)')
    else:
        plt.text(0.5, 0.5, 'Confusion Matrix N/A \n(Sample from validation)',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.axis('off')

    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    out_png = os.path.join(run_dir, f"{plot_name}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"✓ Training curve saved to {out_png}")


def plot_test_metrics(model, test_dataset, run_dir: str, fold_id: int = None,
                      history=None, string: str = None):
    """Plot comprehensive evaluation metrics for test dataset"""
    print("Generating predictions for test metrics...")

    # Collect predictions and labels in batches to avoid memory issues
    y_true_list = []
    y_pred_proba_list = []

    for batch_x, batch_y in test_dataset:
        batch_pred = model.predict(batch_x, verbose=0)
        y_true_list.append(batch_y.numpy())
        y_pred_proba_list.append(batch_pred.flatten())

    y_true = np.concatenate(y_true_list).flatten()
    y_pred_proba = np.concatenate(y_pred_proba_list)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create evaluation plot
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{string} Evaluation Metrics{' (Fold ' + str(fold_id) + ')' if fold_id is not None else ''}",
                 fontsize=16, fontweight='bold')

    # ROC Curve
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Confusion Matrix
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Metrics bar chart
    plt.subplot(2, 2, 3)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': roc_auc}
    bars = plt.bar(range(len(metrics)), list(metrics.values()),
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange'],
                   alpha=0.8, edgecolor='black', linewidth=1)
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Prediction distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Negative Class',
             color='red', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Positive Class',
             color='blue', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    out_png = os.path.join(run_dir, f"{string}_metrics{'_fold' + str(fold_id) if fold_id is not None else ''}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free memory
    print(f"✓ Test metrics visualization saved to {out_png}")

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("=" * 50)

    return {
        'roc_auc': roc_auc, 'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'confusion_matrix': cm.tolist()
    }

def plot_attn(att_model, val_loader, run_dir: str, fold_id: int = None):
    # -------------------------------------------------------------
    # ATTENTION VISUALISATION  – take ONE batch from validation
    # -------------------------------------------------------------
    (pep_ex, mhc_ex), _ = next(iter(val_loader))  # first batch
    att_scores = att_model.predict([pep_ex, mhc_ex], verbose=0)
    print("attn_scores", att_scores.shape)
    print("attn_scores first 5 samples:", att_scores[:5])
    # save attention scores
    if fold_id is None:
        fold_id = 0
    run_dir = os.path.join(run_dir, f"fold_{fold_id}")
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Attention scores saved to {run_dir}")
    out_attn = os.path.join(run_dir, f"attn_scores_fold{fold_id}.npy")
    np.save(out_attn, att_scores)
    print(f"✓ Attention scores saved to {out_attn}")
    # -------------------------------------------------------------
    # att_scores shape : (B, heads, pep_len, mhc_len)
    att_mean = att_scores.mean(axis=1)  # (B,pep,mhc)
    print("att_mean shape:", att_mean.shape)
    sample_id = 0
    A = att_mean[sample_id]
    A = A.transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(A,
    cmap = "viridis",
    xticklabels = [f"P{j}" for j in range(A.shape[1])],
    yticklabels = [f"M{i}" for i in range(A.shape[0])],
    cbar_kws = {"label": "attention"})
    plt.title(f"Fold {fold_id} – attention sample {sample_id}")
    out_png = os.path.join(run_dir,f"attention_fold{fold_id}_sample{sample_id}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Attention heat-map saved to {out_png}")


# ----------------------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True,
                   help="Path to the dataset directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--outdir", default=None,
                   help="Output dir (default: runs/run_YYYYmmdd-HHMMSS)")
    p.add_argument("--buffer_size", type=int, default=1000,
                   help="Buffer size for streaming data loading")
    p.add_argument("--test_batches", type=int, default=3,
                   help="Number of batches to use for test dataset evaluation")

    args = p.parse_args(argv)

    run_dir = args.outdir or f"runs/run_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"★ Outputs → {run_dir}\n")

    # Set seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    print("Setting random seeds for reproducibility...")

    print("Initial memory state:")
    monitor_memory()

    # Extract metadata from datasets without loading them fully
    print("Extracting dataset metadata...")

    # Get fold information
    fold_dir = os.path.join(args.dataset_path, 'folds')
    fold_files = sorted([f for f in os.listdir(fold_dir) if f.endswith('.parquet')])
    n_folds = len(fold_files) // 2

    # Find maximum peptide length across all datasets
    max_peptide_length = 0
    max_mhc_length = 36  # Fixed for now

    print("Scanning datasets for maximum peptide length...")
    all_parquet_files = [
        os.path.join(args.dataset_path, "test1.parquet"),
        os.path.join(args.dataset_path, "test2.parquet")
    ]

    # Add fold files
    for i in range(1, n_folds + 1):
        all_parquet_files.extend([
            os.path.join(fold_dir, f'fold_{i}_train.parquet'),
            os.path.join(fold_dir, f'fold_{i}_val.parquet')
        ])

    for pq_file in all_parquet_files:
        if os.path.exists(pq_file):
            metadata = get_dataset_metadata(pq_file)
            max_peptide_length = max(max_peptide_length, metadata['max_peptide_length'])
            print(
                f"  {os.path.basename(pq_file)}: max_len={metadata['max_peptide_length']}, rows={metadata['total_rows']}")

    print(f"✓ Maximum peptide length across all datasets: {max_peptide_length}")

    # Create fold datasets and class weights
    folds = []
    class_weights = []

    for i in range(1, n_folds + 1):
        print(f"\nProcessing fold {i}/{n_folds}")
        train_path = os.path.join(fold_dir, f'fold_{i}_train.parquet')
        val_path = os.path.join(fold_dir, f'fold_{i}_val.parquet')

        # Calculate class weights from training data
        print(f"  Calculating class weights...")
        cw = calculate_class_weights(train_path)
        print(f"  Class weights: {cw}")

        # Create streaming datasets
        train_ds = (create_streaming_dataset(train_path, max_peptide_length, max_mhc_length,
                                             buffer_size=args.buffer_size)
                    .shuffle(buffer_size=args.buffer_size, reshuffle_each_iteration=True)
                    .batch(args.batch)
                    .take(args.test_batches)
                    .prefetch(tf.data.AUTOTUNE))

        val_ds =   (create_streaming_dataset(val_path, max_peptide_length, max_mhc_length,
                                           buffer_size=args.buffer_size)
                    .batch(args.batch)
                    .take(args.test_batches)
                    .prefetch(tf.data.AUTOTUNE))

        folds.append((train_ds, val_ds))
        class_weights.append(cw)

        # Force cleanup
        cleanup_memory()

    # Create test datasets
    print("Creating test datasets...")
    test1_ds = (create_streaming_dataset(os.path.join(args.dataset_path, "test1.parquet"),
                                         max_peptide_length, max_mhc_length, buffer_size=args.buffer_size)
                .batch(args.batch)
                .prefetch(tf.data.AUTOTUNE))

    test2_ds = (create_streaming_dataset(os.path.join(args.dataset_path, "test2.parquet"),
                                         max_peptide_length, max_mhc_length, buffer_size=args.buffer_size)
                .batch(args.batch)
                .prefetch(tf.data.AUTOTUNE))

    print(f"✓ Created {n_folds} fold datasets and 2 test datasets")
    print("Memory after dataset creation:")
    monitor_memory()

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for fold_id, ((train_loader, val_loader), class_weight) in enumerate(zip(folds, class_weights), start=1):
        print(f'\n🔥 Training fold {fold_id}/{n_folds}')

        # Clean up before each fold
        cleanup_memory()

        # Build fresh model for each fold
        print("Building model...")
        model, attn_model = build_classifier(max_peptide_length,max_mhc_length)
        model.summary()

        # Callbacks
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, f'best_fold_{fold_id}.weights.h5'),
            monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        # Verify data shapes
        for (x_pep, latents), labels in train_loader.take(1):
            print(f"✓ Input shapes: peptide={x_pep.shape}, mhc={latents.shape}, labels={labels.shape}")
            break

        print("Memory before training:")
        monitor_memory()

        # Train model
        print("🚀 Starting training...")
        hist = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=args.epochs,
            class_weight=class_weight,
            callbacks=[ckpt_cb, early_cb],
            verbose=1,
        )

        print("Memory after training:")
        monitor_memory()

        # Plot training curves
        plot_training_curve(hist, run_dir, fold_id, model, val_loader)
        plot_attn(attn_model, val_loader, run_dir, fold_id)

        # Save model and metadata
        model.save_weights(os.path.join(run_dir, f'model_fold_{fold_id}.weights.h5'))
        metadata = {
            "fold_id": fold_id,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "max_peptide_length": max_peptide_length,
            "max_mhc_length": max_mhc_length,
            "class_weights": class_weight,
            "run_dir": run_dir,
            "mhc_class": MHC_CLASS
        }
        with open(os.path.join(run_dir, f'metadata_fold_{fold_id}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        # Evaluate on test sets
        print(f"\n📊 Evaluating fold {fold_id} on test sets...")

        # Test1 evaluation
        print("Evaluating on test1 (balanced alleles)...")
        plot_test_metrics(model, test1_ds, run_dir, fold_id, string="Test1_balanced_alleles")

        # Test2 evaluation
        print("Evaluating on test2 (rare alleles)...")
        plot_test_metrics(model, test2_ds, run_dir, fold_id, string="Test2_rare_alleles")

        print(f"✅ Fold {fold_id} completed successfully")

        # Cleanup
        del model, hist
        cleanup_memory()

    print("\n🎉 Training completed successfully!")
    print(f"📁 All results saved to: {run_dir}")


if __name__ == "__main__":
    BUFFER = 8192  # Reduced buffer size for memory efficiency
    MHC_CLASS = 2
    dataset_path = f"../data/Custom_dataset/NetMHCpan_dataset/mhc_{MHC_CLASS}"
    main([
        "--dataset_path", dataset_path,
        "--epochs", "10",
        "--batch", "32",
        "--buffer_size", "8192",
        "--test_batches", "500",
    ])