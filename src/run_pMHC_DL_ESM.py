#!/usr/bin/env python
"""
mhc_crossattn_pipeline.py
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
    seq_len = int(df["long_mer"].str.len().max())
    print(f"   longest peptide = {seq_len} residues")
    pep_onehot = peptides_to_onehot(df["long_mer"].tolist(), seq_len)
    print("   peptide one-hot shape:", pep_onehot.shape)

    # 2) Latent embeddings --------------------------------------------------
    print("   loading MHC-I embeddings")
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

    return pep_onehot, latents, labels, seq_len


def load_dataset_in_batches(parquet: str, batch_size: int = 100):
    print(f"→ reading {parquet} in batches of {batch_size}")
    df = pd.read_parquet(parquet)
    if "long_mer" not in df.columns:
        raise ValueError("Expected a 'long_mer' column with peptide sequences")
    seq_len = int(df["long_mer"].str.len().max())
    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        pep_onehot = peptides_to_onehot(df["long_mer"].values[start:end].tolist(), seq_len)
        if "mhc_embedding" in df.columns:
            latents = np.stack(df["mhc_embedding"].values[start:end]).astype("float32")
        elif "mhc_embedding_path" in df.columns:
            latents = np.stack([_read_embedding_file(p) for p in tqdm(df["mhc_embedding_path"].values[start:end],
                                                                      desc=f"Loading embeddings {start}-{end}")])
        else:
            raise ValueError("Need a 'mhc_embedding' or 'mhc_embedding_path' column")
        labels = df["assigned_label"].values[start:end].astype("float32")
        yield pep_onehot, latents, labels, seq_len


# ---------------------------------------------------------------------------
# Visualisation utility
# ---------------------------------------------------------------------------
def plot_training_curve(history: tf.keras.callbacks.History, run_dir: str, fold_id: int = None):
    """
    Plot training curves from a Keras history object.
    Args:
        history: Keras history object containing training metrics.
        run_dir: Directory to save the plot.
        fold_id: Optional fold identifier for naming the output file.

    Returns: None

    """
    hist = history.history
    plt.figure(figsize=(21, 4))

    plt.subplot(1, 3, 1)
    plt.plot(hist["loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("BCE loss")
    plt.legend()
    plt.grid(True)

    if "binary_accuracy" in hist and "val_binary_accuracy" in hist:
        plt.subplot(1, 3, 2)
        plt.plot(hist["binary_accuracy"], label="train acc")
        plt.plot(hist["val_binary_accuracy"], label="val acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Binary Accuracy")
        plt.legend()
        plt.grid(True)

    if "auc" in hist and "val_auc" in hist:
        plt.subplot(1, 3, 3)
        plt.plot(hist["auc"], label="train AUC")
        plt.plot(hist["val_auc"], label="val AUC")
        plt.xlabel("epoch")
        plt.ylabel("AUC")
        plt.title("AUC")
        plt.legend()
        plt.grid(True)

    out_png = os.path.join(run_dir, f"training_curve{'_fold' + str(fold_id) if fold_id is not None else ''}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()
    print(f"✓ Training curve saved to {out_png}")


# ---------------------------------------------------------------------------
# TF‑data helper
# ---------------------------------------------------------------------------

def make_tf_dataset(peps, lats, labels, batch=128, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(((peps, lats), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), seed=42)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


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
        batch_loader = load_dataset_in_batches(args.parquet, batch_size=30000)
        peps, lats, labels, seq_len = next(batch_loader)

        print(f"✓ loaded {len(peps):,} samples"
              f" ({peps.shape[1]} residues, {lats.shape[1]} latent features)")

        X_train_p, X_val_p, X_train_l, X_val_l, y_train, y_val = train_test_split(
            peps, lats, labels,
            test_size=args.val_fraction,
            random_state=42,
            stratify=labels)

        train_ds = make_tf_dataset(X_train_p, X_train_l, y_train, batch=args.batch)
        val_ds = make_tf_dataset(X_val_p, X_val_l, y_val, batch=args.batch,
                                 shuffle=False)
    elif args.dataset_path:
        # load test sets
        test1 = pd.read_parquet(args.dataset_path + "/test1.parquet")
        test2 = pd.read_parquet(args.dataset_path + "/test2.parquet")
        num_fold = os.listdir(args.dataset_path + "/folds")
        folds = []
        for i in range(1, len(num_fold) // 2 + 1):
            train_path = args.dataset_path + f"/folds/fold_{i}_train.parquet"
            val_path = args.dataset_path + f"/folds/fold_{i}_val.parquet"
            train_loader = load_dataset_in_batches(train_path, batch_size=3000)
            val_loader = load_dataset_in_batches(val_path, batch_size=3000)

            X_train_p, X_train_l, y_train, seq_len = next(train_loader)
            X_val_p, X_val_l, y_val, _ = next(val_loader)
            train_ds = make_tf_dataset(X_train_p, X_train_l, y_train, batch=args.batch)
            val_ds = make_tf_dataset(X_val_p, X_val_l, y_val, batch=args.batch, shuffle=False)
            folds.append((train_ds, val_ds))

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

    tf.random.set_seed(42);
    np.random.seed(42)
    model = build_classifier(seq_len)
    model.summary()

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(run_dir, "best_weights.h5"),
        monitor="val_loss", save_best_only=True, mode="min")
    early_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)

    if args.parquet:
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=args.epochs,
                            callbacks=[ckpt_cb, early_cb])
        # ----------------------- Plot curve ------------------------------------
        plot_training_curve(history, run_dir)
        # ----------------------- Save model & metadata ------------------------
        meta = dict(parquet=args.parquet,
                    epochs=len(history["loss"]),
                    batch=args.batch,
                    seq_len=seq_len,
                    model_params=dict(embed_dim=256, num_heads=8, ff_dim=512))
        json.dump(meta, open(os.path.join(run_dir, "run_config.json"), "w"), indent=2)
        print("★ Done.")

    elif args.dataset_path:
        # Train on folds
        for i, (train_ds, val_ds) in enumerate(folds):
            print(f"Training on fold {i + 1}/{len(folds)}")
            # clear GPU memory
            tf.keras.backend.clear_session()
            # set random seeds for reproducibility
            tf.random.set_seed(42);
            np.random.seed(42)
            model = build_classifier(seq_len)  # Rebuild model for each fold

            history = model.fit(train_ds,
                                validation_data=val_ds,
                                epochs=args.epochs,
                                callbacks=[ckpt_cb, early_cb])

            # Save fold metadata
            fold_meta = dict(fold_id=i + 1, epochs=len(history.history["loss"]),
                             batch=args.batch, seq_len=seq_len)
            json.dump(fold_meta, open(os.path.join(run_dir, f"fold_{i + 1}_config.json"), "w"), indent=2)

            # ----------------------- Plot training curve ---------------------------
            plot_training_curve(history, run_dir)

        print("★ Done.")
        # TODO think about ensembling folds later

    else:
        raise ValueError("Need either --parquet or --dataset_path argument")


if __name__ == "__main__":
    main([
        # "--parquet", "../data/Custom_dataset/NetMHCpan_dataset/mhc2_with_esm_embeddings.parquet",
        "--dataset_path", "../data/Custom_dataset/NetMHCpan_dataset/mhc_1",
        "--epochs", "100", "--batch", "128"
    ])