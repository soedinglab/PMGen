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

import math
import os, sys, argparse, datetime, pathlib, json
from random import random

print(sys.executable)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model_archive import build_custom_classifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)
import seaborn as sns
import os
import pyarrow.parquet as pq

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


def _load_parquet_rows(parquet_path, seq_len, k=9, test_batches=None):
    pq_file = pq.ParquetFile(parquet_path)
    batch_counter = 0
    for batch_df in pq_file.iter_batches(batch_size=2048):
        if test_batches is not None and batch_counter >= test_batches:
            break
        df = batch_df.to_pandas()

        peps   = tf.constant(df["long_mer"].astype(str).values, dtype=tf.string)
        x_pep  = kmer_onehot_tensor(peps, seq_len, k)  # tf.Tensor on GPU/CPU

        if "mhc_embedding" in df.columns and df["mhc_embedding"].dtype != object:
            latents = tf.constant(np.stack(df["mhc_embedding"]).astype("float32"))
        else:
            latents = tf.constant(np.stack([_read_embedding_file(p) for p in df["mhc_embedding_path"]]))

        labels = tf.constant(df["assigned_label"].astype("float32").values[:, None])
        batch_counter += 1
        yield (x_pep, latents), labels


def make_stream_dataset(
    parquet_path: str,
    max_pep_seq_len: int,
    max_mhc_len: int,
    batch: int = 512,
    shuffle: bool = True,
    k: int = 9,
    test_batches: int = None
) -> tf.data.Dataset:
    """Replacement for the "path‐like" branch; optimized to avoid caching when using small test_batches."""
    rf = max_pep_seq_len - k + 1
    output_signature = (
        (
            tf.TensorSpec(shape=(None, rf, k, 21), dtype=tf.float32), # one-hot peptide tensor
            tf.TensorSpec(shape=(None, max_mhc_len, 1152), dtype=tf.float32), # latent embeddings
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32), # labels
    )

    # Pass test_batches into generator to limit batches early
    ds = tf.data.Dataset.from_generator(
        lambda: _load_parquet_rows(parquet_path, max_pep_seq_len, k, test_batches),
        output_signature=output_signature,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=10 * batch, reshuffle_each_iteration=True)

    # If test_batches is provided, limit dataset and batch after
    if test_batches is not None:
        return ds.prefetch(tf.data.AUTOTUNE)

    # Batch *after* shuffle so that every epoch gets fresh mixes
    return ds.cache().prefetch(tf.data.AUTOTUNE)

# ----------------------------------------------------------------------------
# 4) On‑the‑fly 1:1 class balancing without Pandas down‑sampling
# ----------------------------------------------------------------------------

def build_balanced_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Wrap a dataset so that every step yields a balanced positive/negative mini‑batch."""
    pos = ds.filter(lambda x, y: tf.reduce_any(tf.equal(y, 1)))
    neg = ds.filter(lambda x, y: tf.reduce_any(tf.equal(y, 0)))
    balanced = tf.data.experimental.sample_from_datasets([pos, neg], weights=[0.5, 0.5])
    return balanced

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
    if "auc" in hist and "val_auc" in hist:
        plt.subplot(1, 4, 3)
        plt.plot(hist["auc"], label="train AUC", linewidth=2)
        plt.plot(hist["val_auc"], label="val AUC", linewidth=2)
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
    p.add_argument("--val_fraction", type=float, default=0.2,
                   help="Fraction for train/val split")
    p.add_argument("--test_batches", type=int, default=None,
                   help="Limit to N batches for testing (default: None = use all data)")

    args = p.parse_args(argv)

    run_dir = args.outdir or f"runs/run_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"★ Outputs → {run_dir}\n")

    # ----------------------- Load & split ----------------------------------
    # Load test sets
    test1 = pd.read_parquet(args.dataset_path + "/test1.parquet")
    test2 = pd.read_parquet(args.dataset_path + "/test2.parquet")
    esm_files = list(pathlib.Path(args.dataset_path).glob('*esm_embeddings.parquet'))
    if not esm_files:
        raise FileNotFoundError('No esm embeddings parquet found in dataset_path')
    whole_data = pd.read_parquet(str(esm_files[0]))

    fold_files = sorted([f for f in os.listdir(os.path.join(args.dataset_path, f'folds')) if f.endswith('.parquet')])
    n_folds = len(fold_files) // 2

    # Find the longest peptide sequence across all datasets
    longest_peptide_seq_length = 0
    max_mhc_seq_length = 36 # TODO make dynamic later

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
    # Check all fold files
    for i in range(1, n_folds + 1):
        train_path = os.path.join(args.dataset_path, f'folds/fold_{i}_train.parquet')
        val_path = os.path.join(args.dataset_path, f'folds/fold_{i}_val.parquet')

        train_loader = make_stream_dataset(train_path, max_pep_seq_len=longest_peptide_seq_length, max_mhc_len=max_mhc_seq_length, batch=args.batch, shuffle=True, test_batches=args.test_batches)
        val_loader = make_stream_dataset(val_path, max_pep_seq_len=longest_peptide_seq_length, max_mhc_len=max_mhc_seq_length, batch=args.batch, shuffle=False, test_batches=args.test_batches)

        # # 3) Shuffle + cache + repeat + prefetch for infinite, fresh-each-epoch stream
        # train_ds = (train_loader
        #             .shuffle(buffer_size=100_000, reshuffle_each_iteration=True)
        #             .cache()  # or .cache("train.cache")
        #             .repeat()  # infinite
        #             .prefetch(tf.data.AUTOTUNE))
        #
        # # Validation can skip repeat() and caching if you prefer, but prefetch is still good:
        # val_ds = (val_loader
        #           .shuffle(buffer_size=20_000, reshuffle_each_iteration=True)
        #           .cache()  # optional for val
        #           .prefetch(tf.data.AUTOTUNE))

        # Calculate class weights for the current fold
        train_labels = pd.read_parquet(train_path)["assigned_label"].values.astype(int)
        counts = np.bincount(train_labels, minlength=2)
        cw = {0: 1.0, 1: 1.0}  # Default weights if one class is missing or data is empty
        if counts[0] > 0 and counts[1] > 0:  # If both classes are present
            total = counts.sum()
            cw[0] = total / (2 * counts[0])
            cw[1] = total / (2 * counts[1])

        folds.append((train_loader, val_loader))
        class_weights.append(cw)

    # Create test loaders with the same sequence length
    test1_loader = make_stream_dataset(args.dataset_path + "/test1.parquet"
    , max_pep_seq_len=longest_peptide_seq_length, max_mhc_len=max_mhc_seq_length, batch=128, shuffle=False)
    test2_loader = make_stream_dataset(args.dataset_path + "/test2.parquet"
    , max_pep_seq_len=longest_peptide_seq_length,max_mhc_len=max_mhc_seq_length, batch=128, shuffle=False)

    print(f"✓ loaded {len(test1):,} test1 samples, "
          f"{len(test2):,} test2 samples, "
          f"{len(folds)} folds")

    # ----------------------- Model -----------------------------------------
    # clear GPU memory
    tf.keras.backend.clear_session()
    # set random seeds for reproducibility
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)
    np.random.seed(42)

    # ------------------------- TRAIN --------------------------------------
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(run_dir, 'best_weights.h5'),
        monitor='val_loss', save_best_only=True, mode='min')
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    for fold_id, ((train_loader, val_loader), class_weight) in enumerate(zip(folds, class_weights), start=1):
        print(f'Training on fold {fold_id}/{len(folds)}')
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        print("########################### seq length: ", longest_peptide_seq_length)
        model = build_custom_classifier(longest_peptide_seq_length, max_mhc_seq_length, pad_token=pad_token)
        model.summary()

        # check dimension of the target
        if train_loader.element_spec[1].shape[-1] != 1:
            raise ValueError("Expected binary labels (shape should be (None, 1))")

        print("Training model...")
        hist = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs= args.epochs,
            # class_weight=class_weight,
            callbacks=[ckpt_cb, early_cb],
            verbose=1,
        )

        # plot
        plot_training_curve(hist, run_dir, fold_id, model, val_loader)

        # save model and metadata
        model.save(os.path.join(run_dir, f'model_fold_{fold_id}.h5'))
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
        test1_results = model.evaluate(test1_loader, verbose=1)
        print(f"Test1 results: {test1_results}")
        print("Evaluating on test2 set...")
        test2_results = model.evaluate(test2_loader, verbose=1)
        print(f"Test2 results: {test2_results}")

        # Plot ROC curve for test1
        plot_test_metrics(model, test1_loader, run_dir, fold_id, string="Test1 - balanced alleles")
        # Plot ROC curve for test2
        plot_test_metrics(model, test2_loader, run_dir, fold_id, string="Test2 - rare alleles")


if __name__ == "__main__":
    main([
        "--dataset_path", "../data/Custom_dataset/NetMHCpan_dataset/mhc_1",
        "--epochs", "3", "--batch", "64", "--test_batches", "20"
    ])