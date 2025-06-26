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
AA_TO_INT = {a:i for i,a in enumerate(AA)}
UNK = 20                                                                     # index for “unknown”
PAD_TOKEN = -2 # set manually to avoid confusion with UNK

def onehot(seq: str, max_len: int) -> np.ndarray:
    """Return (max_len,21) one-hot matrix."""
    mat = np.full((max_len, 21), PAD_TOKEN, dtype=np.float32) # initialize padding with -2
    for i, aa in enumerate(seq[:max_len]):
        mat[i, AA_TO_INT.get(aa, UNK)] = 1.0
    return mat

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
                    query_mask=None,                   # (B,pep_len)
                    context_mask=None,                 # (B,mhc_len)
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

    # Create attention mask if both query_mask and context_mask are provided
    if query_mask is not None and context_mask is not None:
        attn_mask = query_mask[:, :, None] & context_mask[:, None, :]  # (B, pep_len, mhc_len)
    else:
        attn_mask = None

    att_out, att_scores = mha(query=query,
                              value=context,
                              key=context,
                              attention_mask=attn_mask,
                              return_attention_scores=True)
    return att_out, att_scores


# ---------------------------------------------------------------------
# main builder
# ---------------------------------------------------------------------
def build_classifier(max_pep_len      : int,
                     max_mhc_len      : int,
                     pep_emb_dim      : int = 64,
                     mhc_emb_dim      : int = 64,
                     heads            : int = 8):
    """
    Returns
        clf_model  – compiled model for training / inference
        att_model  – same weights, outputs attention scores only
    """
    # Inputs -----------------------------------------------------------
    inp_pep = keras.Input(shape=(max_pep_len, 21),    name="pep_onehot")
    inp_mhc = keras.Input(shape=(max_mhc_len,1152),   name="mhc_latent")

    # Masks -----------------------------------------------------------
    pep_mask = layers.Lambda(lambda x: tf.reduce_any(x != PAD_TOKEN, axis=-1), name="pep_mask")(inp_pep)
    mhc_mask = layers.Lambda(lambda x: tf.reduce_any(x != 0, axis=-1), name="mhc_mask")(inp_mhc)

    # Linear projections ----------------------------------------------
    pep_emb = layers.Dense(pep_emb_dim, activation=None,
                           name="pep_proj")(inp_pep)      # (B,pep_len,D)
    mhc_emb = layers.Dense(mhc_emb_dim, activation=None,
                           name="mhc_proj")(inp_mhc)      # (B,mhc_len,D)

    # Positional encoding (simple learned) ----------------------------
    pep_pos = layers.Embedding(input_dim=max_pep_len,
                               output_dim=pep_emb_dim,
                               name="pep_pos_emb")(tf.range(max_pep_len))
    mhc_pos = layers.Embedding(input_dim=max_mhc_len,
                               output_dim=mhc_emb_dim,
                               name="mhc_pos_emb")(tf.range(max_mhc_len))
    pep_emb = pep_emb + pep_pos
    mhc_emb = mhc_emb + mhc_pos

    # Dense layer to reduce dimension of mhc_emb
    mhc_emb = layers.Dense(mhc_emb_dim, activation=None,
                            name="mhc_dim_reduction")(mhc_emb)  # (B,mhc_len,D)

    # Cross-attention --------------------------------------------------
    att_pep, att_scores = cross_att_block(pep_emb, mhc_emb, pep_mask, mhc_mask,
                                          heads=heads, name="pep2mhc")

    # Pool & classifier -----------------------------------------------
    # Mask out padding in peptide for pooling
    masked_att_pep = layers.Lambda(
        lambda inputs: inputs[0] * tf.expand_dims(tf.cast(inputs[1], tf.float32), axis=-1),
        name="mask_pep"
    )([att_pep, pep_mask])
    sum_att_pep = layers.Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name="sum_att_pep"
    )(masked_att_pep)
    num_non_pad = layers.Lambda(
        lambda x: tf.reduce_sum(tf.cast(x, tf.float32), axis=1, keepdims=True),
        name="num_non_pad"
    )(pep_mask)
    pooled = layers.Lambda(
        lambda inputs: inputs[0] / (inputs[1] + 1e-8),
        name="pooled"
    )([sum_att_pep, num_non_pad])
    x      = layers.Dense(32, activation="relu")(pooled)
    out    = layers.Dense(1, activation="sigmoid", name="prob")(x)

    clf_model = keras.Model([inp_pep, inp_mhc], out, name="PepMHC_clf")
    clf_model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["binary_accuracy","AUC"])

    # attention twin ---------------------------------------------------
    att_model = keras.Model([inp_pep, inp_mhc], att_scores,
                            name="PepMHC_attention")

    # cross latent space projection model -----------------------------
    cross_latent_model = keras.Model([inp_pep, inp_mhc], att_pep,
                                        name="PepMHC_cross_latent")

    return clf_model, att_model, cross_latent_model


# --------------------------------------------------------------------------------
# 4. Demo run (synthetic data, 2 epochs, heat map)
# --------------------------------------------------------------------------------
# if __name__ == "__main__":
#     tf.random.set_seed(0);  np.random.seed(0)
#
#     PEP_MAX  = 14
#     MHC_MAX  = 36
#     BATCH    = 64
#     STEPS    = 4              # keep tiny for a quick sanity run
#
#     model, att_model, cross_latent_model = build_classifier(max_pep_len=PEP_MAX, max_mhc_len=MHC_MAX)
#
#     print(model.summary(line_length=110))
#
#     # quick dummy training --------------------------------------------------
#     for step in range(STEPS):
#         Xp, Xm, y = toy_batch_masked(BATCH, pep_max=PEP_MAX, mhc_max=MHC_MAX)
#         model.train_on_batch([Xp,Xm], y)
#
#     # one inference batch & attention ---------------------------------------
#     Xp_test, Xm_test, y_test = toy_batch_masked(8, pep_max=PEP_MAX, mhc_max=MHC_MAX)
#     preds      = model.predict([Xp_test, Xm_test])
#     att_scores = att_model.predict([Xp_test, Xm_test])        # (B,heads,pep_len,mhc_len)
#
#     print("\nPredictions:", preds.squeeze().round(3))
#
#     # --------------------------------------------------------------------------------
#     # 5. visualise attention for the first sample (average over heads)
#     # --------------------------------------------------------------------------------
#     sample   = 0
#     A = att_scores[sample]                     # (heads,pep_len,mhc_len)
#     A_mean = A.mean(axis=0)                   # (pep_len,mhc_len)
#
#     plt.figure(figsize=(8,6))
#     sns.heatmap(A_mean,
#                 cmap="viridis",
#                 xticklabels=[f"M{i}" for i in range(MHC_MAX)],
#                 yticklabels=[f"P{j}" for j in range(PEP_MAX)],
#                 cbar_kws={"label":"attention"})
#     plt.xlabel("MHC position");  plt.ylabel("Peptide position")
#     plt.title("Peptide→MHC attention (heads averaged) – sample 0")
#     plt.tight_layout()
#     plt.show()
#
#     # report positions with highest influence ------------------------------
#     pep_pos = np.argmax(A_mean, axis=1)              # best MHC pos for each peptide token
#     mhc_pos = np.argmax(A_mean, axis=0)              # most queried peptide pos per MHC token
#
#     print("\nTop MHC position attended by each peptide residue:")
#     for p in range(PEP_MAX):
#         print(f"  peptide P{p:02d} ⇢ MHC M{pep_pos[p]:02d}  (score={A_mean[p,pep_pos[p]]:.3f})")
#
#     print("\nPeptide position with max attention received from each MHC residue:")
#     for m in range(MHC_MAX):
#         print(f"  MHC M{m:02d} ⇠ peptide P{mhc_pos[m]:02d} (score={A_mean[mhc_pos[m],m]:.3f})")
