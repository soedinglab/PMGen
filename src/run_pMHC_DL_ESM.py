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
import os, sys, argparse, datetime, pathlib, json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.model import build_classifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)
import seaborn as sns


# ---------------------------------------------------------------------------
# Utility: peptide → one‑hot (seq_len, 21)
# ---------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard AAs, order fixed
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
UNK_IDX = 20  # index for unknown / padding


def peptides_to_onehot(seqs: list[str], seq_len: int) -> np.ndarray:
    """Vectorise *seqs* into (N, seq_len, 21) one‑hot with padding."""
    N = len(seqs)
    arr = np.zeros((N, seq_len, 21), dtype=np.float32)
    for i, s in enumerate(seqs):
        if not s:
            raise "[ERR] empty peptide sequence"
        for j, aa in enumerate(s.upper()[:seq_len]):
            arr[i, j, AA_TO_IDX.get(aa, UNK_IDX)] = 1.0
        # remaining positions stay zero → acts as padded UNK (index 20)
    return arr


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


# ---------------------------------------------------------------------------
# Dataset loader – returns (peptide_onehot, latent), labels
# ---------------------------------------------------------------------------

def load_dataset(parquet_path: str):
    print(f"→ Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 1) Peptide one‑hot ----------------------------------------------------
    if "long_mer" not in df.columns:
        raise ValueError("Expected a 'long_mer' column with peptide sequences")
    pep_seq_len = int(df["long_mer"].str.len().max())
    print(f"   longest peptide = {pep_seq_len} residues")
    pep_onehot = peptides_to_onehot(df["long_mer"].tolist(), pep_seq_len)
    print("   peptide one-hot shape:", pep_onehot.shape)

    # 2) Latent embeddings --------------------------------------------------
    print(f"   loading MHC embeddings")
    if "mhc_embedding" in df.columns:
        latents = np.stack(df["mhc_embedding"].values).astype("float32")
    elif "mhc_embedding_path" in df.columns:
        latents = np.stack([_read_embedding_file(p) for p in df["mhc_embedding_path"]])
    else:
        raise ValueError("Need 'mhc_embedding' or 'mhc_embedding_path' column")

    print("   latent shape:", latents.shape)
    if latents.shape[1:] != (36, 1152):
        raise ValueError(f"Unexpected latent shape {latents.shape[1:]}, expected (36,1152)")

    # 3) Labels -------------------------------------------------------------
    labels = df["assigned_label"].astype("float32").values[:, None]  # (N,1)

    return pep_onehot, latents, labels, pep_seq_len


def load_dataset_in_batches(parquet: str, target_seq_len: int, batch_size: int = 100, subset: float = None):
    print(f"→ reading {parquet} in batches of {batch_size}")
    if subset is not None:
        print(f"   using subset fraction: {subset}")
        df = pd.read_parquet(parquet).sample(frac=subset, random_state=42)
    else:
        df = pd.read_parquet(parquet)
    if "long_mer" not in df.columns:
        raise ValueError("Expected a 'long_mer' column with peptide sequences")
    # REMOVE or IGNORE: pep_seq_len = int(df["long_mer"].str.len().max()) # This was the problematic line for one-hot encoding
    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        # Use target_seq_len for one-hot encoding
        pep_onehot = peptides_to_onehot(df["long_mer"].values[start:end].tolist(), target_seq_len)
        if "mhc_embedding" in df.columns:
            latents = np.stack(df["mhc_embedding"].values[start:end]).astype("float32")
        elif "mhc_embedding_path" in df.columns:
            latents = np.stack([_read_embedding_file(p) for p in df["mhc_embedding_path"].values[start:end]])
        else:
            raise ValueError("Need a 'mhc_embedding' or 'mhc_embedding_path' column")

        if latents.shape[1:] != (36, 1152):
            raise ValueError(f"Unexpected latent shape {latents.shape[1:]}, expected (36,1152)")

        labels = df["assigned_label"].values[start:end].astype("int32")
        labels = labels[:, None]

        yield pep_onehot, latents, labels # Removed the local pep_seq_len from yield



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
# TF‑data helper
# ---------------------------------------------------------------------------

def make_tf_dataset(source, longest_peptide_seq_length, batch: int = 128, shuffle: bool = True, subset: float = None):
    if isinstance(source, (str, os.PathLike)):
        def gen():
            # Pass longest_peptide_seq_length as target_seq_len
            # The generator now yields 3 items
            for peps, lats, labs in load_dataset_in_batches(str(source),
                                                            target_seq_len=longest_peptide_seq_length,
                                                            batch_size=batch,
                                                            subset=subset):
                yield (peps, lats), labs
        output_signature = (
            (tf.TensorSpec(shape=(None, longest_peptide_seq_length, 21), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 36, 1152), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            # Consider increasing buffer_size for better shuffling if dataset is large
            # e.g., buffer_size=num_batches_in_epoch or a sufficiently large number
            ds = ds.shuffle(buffer_size=max(10, len(pd.read_parquet(str(source))) // batch // 2 ), seed=42)
        return ds.prefetch(tf.data.AUTOTUNE)
    else: # This part for in-memory dataframes seems correct as it uses longest_peptide_seq_length
        peps = peptides_to_onehot(source["long_mer"].tolist(), longest_peptide_seq_length)
        labels = source["assigned_label"].values.astype("float32")[:, None]

        if "mhc_embedding" in source.columns:
            lats = np.stack(source["mhc_embedding"].values).astype("float32")
        elif "mhc_embedding_path" in source.columns:
            lats = np.stack([_read_embedding_file(p) for p in source["mhc_embedding_path"]]).astype("float32")
        else:
            raise ValueError("Need 'mhc_embedding' or 'mhc_embedding_path' column")

        ds = tf.data.Dataset.from_tensor_slices(((peps, lats), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(labels), seed=42)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    # peps = source["long_mer"]
    # labels = source["assigned_label"]
    # lats = np.stack([np.load(p) for p in source["mhc_embedding_path"]]).astype("float32")
    # ds = tf.data.Dataset.from_tensor_slices(((peps, lats), labels))
    # if shuffle:
    #     ds = ds.shuffle(buffer_size=len(labels), seed=42)
    # return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# def _collect_batches(loader, max_batch_size: int = 3_000):
#     '''
#     Concatenate a lazy *loader* (yielding pep, latent, label, seq_len)
#     into single large NumPy arrays **with just one final copy**.
#
#     On each iteration we append references to the individual batch arrays;
#     only at the very end do we *concatenate* along axis0.  This strategy is
#     RAM‑friendly because the intermediate lists only hold *views* of the
#     already‑allocated batch memory while the batches are alive, and we avoid
#     the O(N²) cost of repeated `np.concatenate` calls inside the loop.
#
#     If the total number of samples reaches *max_batch_size*, stop loading more.
#     '''
#     pep_chunks, lat_chunks, lab_chunks = [], [], []
#     seq_len = None
#     total = 0
#
#     for peps, lats, labels, seq_len in loader:
#         n = len(peps)
#         if total + n > max_batch_size:
#             # Only take up to the limit
#             take = max_batch_size - total
#             if take <= 0:
#                 break
#             pep_chunks.append(peps[:take])
#             lat_chunks.append(lats[:take])
#             labels = labels[:take]
#             labels = labels.reshape(-1, 1) if labels.ndim == 1 else labels
#             lab_chunks.append(labels)
#             total += take
#             break
#         pep_chunks.append(peps)
#         lat_chunks.append(lats)
#         labels = labels.reshape(-1, 1) if labels.ndim == 1 else labels
#         lab_chunks.append(labels)
#         total += n
#         if total >= max_batch_size:
#             break
#
#     # single allocation per tensor ↓
#     peps   = np.concatenate(pep_chunks, axis=0, dtype=np.float32)
#     lats   = np.concatenate(lat_chunks, axis=0, dtype=np.float32)
#     labels = np.concatenate(lab_chunks, axis=0, dtype=np.float32)
#
#     # explicitly free the now‑unneeded small arrays
#     del pep_chunks, lat_chunks, lab_chunks
#     return peps, lats, labels, seq_len

# convenience wrappers -------------------------------------------------------

# def collect_parquet(parquet_path: str, batch_size: int = 30_000):
#     '''Load **all** samples in *parquet_path* using streaming batches.'''
#     loader = load_dataset_in_batches(parquet_path, batch_size=batch_size)
#     return _collect_batches(loader)
#
#
# def collect_fold(fold_path: str, batch_size: int = 3_000):
#     return collect_parquet(fold_path, batch_size=batch_size)

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
    args = p.parse_args(argv)

    run_dir = args.outdir or f"runs/run_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
    print(f"★ Outputs → {run_dir}\n")

    # ----------------------- Load & split ----------------------------------
    if args.parquet:
        # peps, lats, labels, seq_len = load_dataset(args.parquet)
        peps, lats, labels, longest_peptide_seq_length = load_dataset(args.parquet)

        print(f"✓ loaded {len(peps):,} samples"
              f" ({peps.shape[1]} residues, {lats.shape[1]} latent features)")

        X_train_p, X_val_p, X_train_l, X_val_l, y_train, y_val = train_test_split(
            peps, lats, labels,
            test_size=args.val_fraction,
            random_state=42,
            stratify=labels)

        train_loader =  make_tf_dataset((X_train_p, X_train_l, y_train), longest_peptide_seq_length=longest_peptide_seq_length, batch=args.batch, shuffle=True)
        val_loader = make_tf_dataset((X_val_p, X_val_l, y_val), longest_peptide_seq_length=longest_peptide_seq_length ,batch=args.batch, shuffle=False)

    elif args.dataset_path:
        # Load test sets
        test1 = pd.read_parquet(args.dataset_path + "/test1.parquet")
        test2 = pd.read_parquet(args.dataset_path + "/test2.parquet")
        fold_files = sorted([f for f in os.listdir(os.path.join(args.dataset_path, 'folds')) if f.endswith('.parquet')])
        n_folds = len(fold_files) // 2

        # Find the longest peptide sequence across all datasets
        longest_peptide_seq_length = 0

        # Check test datasets
        if "long_mer" in test1.columns:
            longest_peptide_seq_length = max(longest_peptide_seq_length, int(test1["long_mer"].str.len().max()))
        if "long_mer" in test2.columns:
            longest_peptide_seq_length = max(longest_peptide_seq_length, int(test2["long_mer"].str.len().max()))

        # Check all fold files
        for i in range(1, n_folds + 1):
            train_path = os.path.join(args.dataset_path, f'folds/fold_{i}_train.parquet')
            val_path = os.path.join(args.dataset_path, f'folds/fold_{i}_val.parquet')

            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)

            if "long_mer" in train_df.columns:
                longest_peptide_seq_length = max(longest_peptide_seq_length, int(train_df["long_mer"].str.len().max()))
            if "long_mer" in val_df.columns:
                longest_peptide_seq_length = max(longest_peptide_seq_length, int(val_df["long_mer"].str.len().max()))

        print(f"✓ Longest peptide sequence length across all datasets: {longest_peptide_seq_length}")

        # Create fold datasets with consistent sequence length
        folds = []
        for i in range(1, n_folds + 1):
            train_path = os.path.join(args.dataset_path, f'folds/fold_{i}_train.parquet')
            val_path = os.path.join(args.dataset_path, f'folds/fold_{i}_val.parquet')

            train_loader = make_tf_dataset(train_path, longest_peptide_seq_length=longest_peptide_seq_length, batch=args.batch, shuffle=True)
            val_loader = make_tf_dataset(val_path, longest_peptide_seq_length=longest_peptide_seq_length, batch=args.batch, shuffle=False)

            folds.append((train_loader, val_loader))

        # Create test loaders with the same sequence length
        test1_loader = make_tf_dataset(test1, longest_peptide_seq_length=longest_peptide_seq_length, batch=args.batch, shuffle=False)
        test2_loader = make_tf_dataset(test2, longest_peptide_seq_length=longest_peptide_seq_length, batch=args.batch, shuffle=False)

        print(f"✓ loaded {len(test1):,} test1 samples, "
              f"{len(test2):,} test2 samples, "
              f"{len(folds)} folds")
    else:
        raise ValueError("Need either --parquet or --dataset_path argument")

    # ----------------------- Model -----------------------------------------
    # clear GPU memory
    tf.keras.backend.clear_session()
    # set random seeds for reproducibility
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)
    np.random.seed(42)

        # # Verify and explicitly pad/trim to match training seq_len
        # if test1_seq_len != seq_len or test2_seq_len != seq_len:
        #     print(f"Adjusting test datasets to match training seq_len: {seq_len}")
        #     X_test1_p = X_test1_p[:, :seq_len, :]
        #     X_test2_p = X_test2_p[:, :seq_len, :]
        #
        # # Pad or trim test peptides to match seq_len
        # X_test1_p = X_test1_p[:, :seq_len, :]
        # X_test2_p = X_test2_p[:, :seq_len, :]
        # test1_ds = make_tf_dataset(X_test1_p, X_test1_l, y_test1, batch=args.batch, shuffle=False)
        # test2_ds = make_tf_dataset(X_test2_p, X_test2_l, y_test2, batch=args.batch, shuffle=False)

    #     print(f'✓ loaded {len(test1):,} test1 samples, '
    #           f'{len(test2):,} test2 samples, '
    #           f'{len(folds)} folds')
    #
    #     print("★ Done.")
    #     # TODO think about ensembling folds later
    #
    # else:
    #     raise ValueError("Need either --parquet or --dataset_path argument")

    # ------------------------- TRAIN --------------------------------------
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(run_dir, 'best_weights.h5'),
        monitor='val_loss', save_best_only=True, mode='min')
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    if args.parquet:
        tf.keras.backend.clear_session()
        os.environ['PYTHONHASHSEED'] = '42'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        model = build_classifier(longest_peptide_seq_length)
        model.summary()


        history = model.fit(train_loader,
                            validation_data=val_loader,
                            epochs=args.epochs,
                            callbacks=[ckpt_cb, early_cb])

        # plot
        plot_training_curve(history, run_dir, fold_id=None, model=model, val_dataset=val_loader)

        # save model and metadata
        model.save(os.path.join(run_dir, 'model.h5'))
        metadata = {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "longest_peptide_seq_length": longest_peptide_seq_length,
            "run_dir": run_dir
        }
        with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)


    elif args.dataset_path:
        for fold_id, (train_loader, val_loader) in enumerate(folds, start=1):
            print(f'Training on fold {fold_id}/{len(folds)}')
            tf.keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)
            print("########################### seq length: ", longest_peptide_seq_length)
            model = build_classifier(longest_peptide_seq_length)
            model.summary()

            # print one sample
            # print("Sample input shape:", next(iter(train_loader))[0][0].shape)
            # print("Sample latent shape:", next(iter(train_loader))[0][1].shape)
            # print("Sample label shape:", next(iter(train_loader))[1].shape)

            history = model.fit(train_loader,
                                validation_data=val_loader,
                                epochs=args.epochs,
                                callbacks=[ckpt_cb, early_cb])

            # plot
            plot_training_curve(history, run_dir, fold_id, model, val_loader)

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
        # "--parquet", "../data/Custom_dataset/NetMHCpan_dataset/mhc_2/mhc2_with_esm_embeddings.parquet",
        "--dataset_path", "../data/Custom_dataset/NetMHCpan_dataset/mhc_2",
        "--epochs", "100", "--batch", "128"
    ])