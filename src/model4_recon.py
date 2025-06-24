#!/usr/bin/env python
"""
pepmhc_cross_attention.py
-------------------------

Minimal end-to-end demo of a peptide × MHC cross-attention classifier
with explainable attention visualisation.

Author: 2025-05-22
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------
# 1. Synthetic toy-data helpers
# --------------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
MASK_IDX = 20  # new index
AA_TO_INT = {a: i for i, a in enumerate(AA)}
AA_DIM = 21  # 20 AA + 1 MASK

def onehot(seq: str, max_len: int) -> np.ndarray:
    mat = np.zeros((max_len, AA_DIM), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        mat[i, AA_TO_INT.get(aa, MASK_IDX)] = 1.0
    return mat

def onehot_to_seq(onehot_mat: np.ndarray) -> str:
    """Convert one-hot encoding back to amino acid sequence."""
    indices = np.argmax(onehot_mat, axis=1)
    seq = ""
    for idx in indices:
        if idx == MASK_IDX:
            seq += "X"  # Use X to represent masked positions
        else:
            seq += AA[idx]
    return seq

def toy_batch_masked(batch_size=32,
                     pep_max=14,
                     mhc_max=36,
                     mask_rate=0.3,  # 30\% of peptide tokens will be masked
                     mhc_dim=1152):
    """Synthetic (masked peptide input, true peptide, mask weights, mhc) batch."""
    peps_in, peps_true, mask_w, mhcs = [], [], [], []
    for _ in range(batch_size):
        # ----- peptide -------------------------------------------------------
        Lp = np.random.randint(8, pep_max + 1)
        pep = ''.join(np.random.choice(list(AA), Lp))
        oh = onehot(pep, pep_max)  # ground-truth (B,pep_max,21)

        # decide which positions to mask
        mpos = np.random.rand(pep_max) < mask_rate
        oh_masked = oh.copy()
        oh_masked[mpos] = 0.0
        oh_masked[mpos, MASK_IDX] = 1.0  # replace by MASK token

        peps_in.append(oh_masked)  # model input
        peps_true.append(oh)  # y_true
        mask_w.append(mpos.astype(np.float32))  # sample-weight
        # ----- MHC latent ----------------------------------------------------
        Lm = np.random.randint(20, mhc_max + 1)
        mhc_lat = np.random.randn(Lm, mhc_dim).astype(np.float32)
        pad_mhc = np.zeros((mhc_max, mhc_dim), np.float32)
        pad_mhc[:Lm] = mhc_lat
        mhcs.append(pad_mhc)

    return (np.stack(peps_in),
            np.stack(mhcs),
            np.stack(peps_true),
            np.stack(mask_w))


# --------------------------------------------------------------------------------
# 2. Model building block: cross-attention with score output
# --------------------------------------------------------------------------------
"""
model.py  –  light-weight peptide × MHC cross-attention classifier
             plus twin model that exposes the attention matrix
             for explainability.

The only public symbol is
    build_classifier(max_pep_len, max_mhc_len, …)
which returns
    clf_model, att_model
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def cross_att_block(query,                              # (B,pep_len,D)
                    context,                            # (B,mhc_len,D)
                    heads=8,
                    name="xatt"):
    """
    Keras MultiHeadAttention; returns   (att_out, att_scores)
    Shapes:
        att_out     (B, pep_len, D)
        att_scores  (B, heads, pep_len, mhc_len)
    """
    mha = layers.MultiHeadAttention(num_heads=heads,
                                    key_dim=query.shape[-1],
                                    name=name)

    att_out, att_scores = mha(query=query,
                              value=context,
                              key=context,
                              return_attention_scores=True)
    return att_out, att_scores


# ---------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------
# ----- inputs ------------------------------------------------------------


def build_reconstruction_model(max_pep_len      : int,
                                 max_mhc_len      : int,
                                 pep_emb_dim      : int = 64,
                                 mhc_emb_dim      : int = 64,
                                 mhc_latent_dim   : int = 1152,
                                 heads            : int = 8):
    """
    Returns
        clf_model  – compiled model for training / inference
        att_model  – same weights, outputs attention scores only
    """
    # Inputs -----------------------------------------------------------
    inp_pep = keras.Input(shape=(max_pep_len, AA_DIM),    name="pep_onehot")
    inp_mhc = keras.Input(shape=(max_mhc_len,mhc_latent_dim),   name="mhc_latent")

    # ----- linear projections & positional enc ------------------------------
    pep_emb = layers.Dense(pep_emb_dim, activation=None, name="pep_proj")(inp_pep)
    mhc_emb = layers.Dense(mhc_emb_dim, activation=None, name="mhc_proj")(inp_mhc)

    pep_pos = layers.Embedding(max_pep_len, pep_emb_dim, name="pep_pos")(tf.range(max_pep_len))
    mhc_pos = layers.Embedding(max_mhc_len, mhc_emb_dim, name="mhc_pos")(tf.range(max_mhc_len))
    pep_emb = pep_emb + pep_pos
    mhc_emb = mhc_emb + mhc_pos
    mhc_emb = layers.Dense(mhc_emb_dim, activation=None, name="mhc_dim_reduction")(mhc_emb)

    # ----- cross-attention ---------------------------------------------------
    att_pep, att_scores = cross_att_block(pep_emb, mhc_emb, heads=heads, name="pep2mhc")

    # ----- per-token classifier ---------------------------------------------
    logits = layers.Dense(AA_DIM, activation=None, name="logits")(att_pep)
    probs = layers.Activation("softmax", name="aa_probs")(logits)  # (B,pep_len,21)

    model = keras.Model([inp_pep, inp_mhc], logits, name="PepMHC_maskedLM")

    model.compile(
        optimizer="adam",
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="masked_accuracy")],
        sample_weight_mode="temporal",
    )

    # optional twin that only outputs attention ------------------------------
    att_model = keras.Model([inp_pep, inp_mhc], att_scores, name="PepMHC_attention")

    return model, att_model


# --------------------------------------------------------------------------------
# 4. Demo run (synthetic data, 2 epochs, heat map)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    tf.random.set_seed(0);  np.random.seed(0)

    PEP_MAX  = 14
    MHC_MAX  = 36
    BATCH    = 64
    STEPS    = 40              # keep tiny for a quick sanity run

    model, att_model = build_reconstruction_model(max_pep_len=PEP_MAX, max_mhc_len=MHC_MAX)

    print(model.summary(line_length=110))

    # quick dummy training --------------------------------------------------
    for step in range(STEPS):
        Xp, Xm, y_true, w = toy_batch_masked(BATCH, pep_max=PEP_MAX, mhc_max=MHC_MAX)
        model.train_on_batch([Xp, Xm], y_true, sample_weight=w)

    ##############
    Xp_test, Xm_test, y_test, mask_w_test = toy_batch_masked(8, pep_max=PEP_MAX, mhc_max=MHC_MAX)
    preds = model.predict([Xp_test, Xm_test])
    att_scores = att_model.predict([Xp_test, Xm_test])  # (B,heads,pep_len,mhc_len)

    # Visualize peptide reconstruction examples
    print("\n=== Peptide Reconstruction Examples ===")
    for i in range(8):  # Show first 3 examples
        # Get original, masked and reconstructed peptides
        original_pep = onehot_to_seq(y_test[i])
        masked_pep = onehot_to_seq(Xp_test[i])
        recon_pep = onehot_to_seq(preds[i])

        # Trim to actual peptide length (remove trailing padding)
        actual_len = len(original_pep.rstrip())
        original_pep = original_pep[:actual_len]
        masked_pep = masked_pep[:actual_len]
        recon_pep = recon_pep[:actual_len]

        # Highlight differences with formatting
        highlighted_recon = ""
        for j, (o, r) in enumerate(zip(original_pep, recon_pep)):
            if masked_pep[j] == "X":
                if o == r:
                    highlighted_recon += f"[{r}]"  # Correctly reconstructed (was masked)
                else:
                    highlighted_recon += f"({r})"  # Incorrectly reconstructed (was masked)
            else:
                highlighted_recon += r  # Position was not masked

        print(f"Example {i + 1}:")
        print(f"  Original:     {original_pep}")
        print(f"  Masked input: {masked_pep}")
        print(f"  Reconstructed: {highlighted_recon}")
        print(f"  ([correct] / (incorrect) reconstruction of masked positions)\n")

    # Visualize attention for one example
    plt.figure(figsize=(12, 5))
    example_idx = 0
    att_example = np.mean(att_scores[example_idx], axis=0)  # Average over attention heads

    # Get non-padding length
    pep_len = len(onehot_to_seq(y_test[example_idx]).rstrip())
    mhc_len = np.sum(np.any(Xm_test[example_idx] != 0, axis=1))

    # Only display the actual peptide and MHC lengths (not padding)
    att_display = att_example[:pep_len, :mhc_len]

    # Create heatmap
    ax = sns.heatmap(att_display, cmap='viridis')
    plt.title(f"Peptide-MHC Cross-Attention (Example {example_idx + 1})")
    plt.xlabel("MHC position")
    plt.ylabel("Peptide position")

    # Annotate masked positions on y-axis
    masked_pep = onehot_to_seq(Xp_test[example_idx])[:pep_len]
    plt.yticks(np.arange(pep_len) + 0.5,
               [f"{i}:{aa}" + ("*" if aa == "X" else "")
                for i, aa in enumerate(masked_pep)],
               rotation=0)

    plt.tight_layout()
    plt.show()


