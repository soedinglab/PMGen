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
from random import random

print(sys.executable)

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ~8×-40× faster on CPU, zero-copy on GPU
import tensorflow as tf

AA = tf.constant(list("ACDEFGHIKLMNPQRSTVWY"))
TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(AA,
                                        tf.range(20, dtype=tf.int32)),
    default_value=20)                 # UNK_IDX

def peptides_to_onehot_tf(seqs, seq_len, k=9):
    # seqs : tf.Tensor([b'PEPTIDE', ...])  shape=(N,)
    # output: (N, RF, k, 21) where RF = seq_len - k + 1
    tokens = tf.strings.bytes_split(seqs)                       # ragged ‹N, L›
    idx    = TABLE.lookup(tokens.flat_values)
    ragged = tf.RaggedTensor.from_row_lengths(idx, tokens.row_lengths())
    idx_pad = ragged.to_tensor(default_value=20, shape=(None, seq_len))
    onehot  = tf.one_hot(idx_pad, 21, dtype=tf.float32)         # (N, seq_len, 21)

    RF = seq_len - k + 1
    ta = tf.TensorArray(dtype=tf.float32, size=RF)
    def body(i, ta):
        slice_k = onehot[:, i:i+k, :]                           # (N, k, 21)
        return i+1, ta.write(i, slice_k)
    _, ta = tf.while_loop(lambda i, _: i < RF, body, [0, ta])
    patches = ta.stack()                                        # (RF, N, k, 21)
    return tf.transpose(patches, [1, 0, 2, 3])                  # (N, RF, k, 21)


class AttentionLayer(keras.layers.Layer):
    """
    Custom multi-head attention layer supporting self- and cross-attention.

    Args:
        input_dim (int): Input feature dimension.
        output_dim (int): Output feature dimension per head.
        type (str): 'self' or 'cross'.
        heads (int): Number of attention heads.
        resnet (bool): Whether to use residual connection.
        return_att_weights (bool): Whether to return attention weights.
        name (str): Name for weight scopes.
        epsilon (float): Epsilon for layer normalization.
        gate (bool): Whether to use gating mechanism.
    """

    def __init__(self, input_dim, output_dim, type, heads=4,
                 resnet=True, return_att_weights=False, name='attention',
                 epsilon=1e-6, gate=True, mask_token=-1., pad_token=-2.):
        super().__init__(name=name)
        assert isinstance(input_dim, int) and isinstance(output_dim, int)
        assert type in ['self', 'cross']
        if resnet:
            assert input_dim == output_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type = type
        self.heads = heads
        self.resnet = resnet
        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.gate = gate
        self.mask_token = mask_token
        self.pad_token = pad_token

    def build(self, x):
        self.q = self.add_weight(shape=(self.heads, self.input_dim, self.output_dim),
                                 initializer='random_normal', trainable=True, name=f'q_{self.name}')
        self.k = self.add_weight(shape=(self.heads, self.input_dim, self.output_dim),
                                 initializer='random_normal', trainable=True, name=f'k_{self.name}')
        self.v = self.add_weight(shape=(self.heads, self.input_dim, self.output_dim),
                                 initializer='random_normal', trainable=True, name=f'v_{self.name}')
        if self.gate:
            self.g = self.add_weight(shape=(self.heads, self.input_dim, self.output_dim),
                                     initializer='random_uniform', trainable=True, name=f'gate_{self.name}')
        self.norm = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_{self.name}')
        if self.type == 'cross':
            self.norm_context = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_context_{self.name}')
        self.norm_out = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_out_{self.name}')
        if self.resnet:
            self.norm_resnet = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_resnet_{self.name}')
        self.out_w = self.add_weight(shape=(self.output_dim * self.heads, self.output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{self.name}')
        self.out_b = self.add_weight(shape=(self.output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{self.name}')
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.output_dim, tf.float32))

    def call(self, x, mask, context=None, context_mask=None):
        """
        Args:
            x: Tensor of shape (B, N, D)
            mask: Tensor of shape (B,N)
            context: Optional tensor (B, M, D) for cross-attention
        """
        # Auto-generate padding mask if not provided (based on all-zero tokens)
        mask = tf.cast(mask, tf.float32)  # shape: (B, N)

        x_norm = self.norm(x)
        if self.type == 'self':
            q_input = k_input = v_input = x_norm
            mask = tf.where(mask == self.pad_token, 0.,
                            1.)  # all padded ones are 0 and masked ones (for mask learning) and normal ones are 1
            mask_k = mask_q = mask
        else:
            assert context is not None, "context is required for cross-attention"
            assert context_mask is not None, "context_mask is required for cross-attention"
            context_norm = self.norm_context(context)
            q_input = x_norm
            k_input = v_input = context_norm
            mask_q = tf.where(mask == self.pad_token, 0., 1.)
            mask_k = tf.where(context_mask == self.pad_token, 0., 1.)

        q = tf.einsum('bnd,hde->hbne', q_input, self.q)
        k = tf.einsum('bmd,hde->hbme', k_input, self.k)
        v = tf.einsum('bmd,hde->hbme', v_input, self.v)

        att = tf.einsum('hbne,hbme->hbnm', q, k) * self.scale

        # Add large negative mask to padded keys
        mask_k = tf.expand_dims(mask_k, 1)  # (B, 1, M)
        mask_q = tf.expand_dims(mask_q, 1)  # (B, 1, N)
        attention_mask = tf.einsum('bqn,bkm->bnm', mask_q, mask_k)  # (B, N, M)
        attention_mask = tf.expand_dims(attention_mask, 0)  # (1, B, N, M)
        att += (1.0 - attention_mask) * -1e9

        att = tf.nn.softmax(att, axis=-1) * attention_mask

        out = tf.einsum('hbnm,hbme->hbne', att, v)

        if self.gate:
            g = tf.einsum('bnd,hde->hbne', x_norm, self.g)
            g = tf.nn.sigmoid(g)
            out *= g

        if self.resnet:
            out += tf.expand_dims(x, axis=0)
            out = self.norm_resnet(out)

        out = tf.transpose(out, [1, 2, 3, 0])  # (B, N, E, H)
        out = tf.reshape(out, [tf.shape(x)[0], tf.shape(x)[1], self.output_dim * self.heads])
        out = tf.matmul(out, self.out_w) + self.out_b

        if self.resnet:
            out += x
        out = self.norm_out(out)
        # Zero out padded tokens after bias addition
        mask_exp = tf.expand_dims(mask, axis=-1)  # (B, N, 1)
        out *= mask_exp
        return (out, att) if self.return_att_weights else out


class PositionalEncoding(keras.layers.Layer):
    """
    Sinusoidal Positional Encoding layer that applies encodings
    only to non-masked tokens.

    Args:
        embed_dim (int): Dimension of embeddings (must match input last dim).
        max_len (int): Maximum sequence length expected (used to precompute encodings).
    """

    def __init__(self, embed_dim, max_len: int =100, mask_token: float =-1., pad_token: float =-2., name: str ='positional_encoding'):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.mask_token = mask_token
        self._name = name
        self.pad_token = pad_token

    def build(self, x):
        # Create (1, max_len, embed_dim) encoding matrix
        pos = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]  # (max_len, 1)
        i = tf.range(self.embed_dim, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim)
        angle_rates = 1 / tf.pow(300.0, (2 * (i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = pos * angle_rates  # (max_len, embed_dim)

        # Apply sin to even indices, cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)  # (max_len, embed_dim)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, max_len, embed_dim)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B,N)
        Returns:
            Tensor with positional encodings added for masked and non padded tokens.
        """
        seq_len = tf.shape(x)[1]
        pe = self.pos_encoding[:, :seq_len, :]  # (1, N, D)
        mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)  # (B, N, 1)
        mask = tf.where(mask == self.pad_token, 0., 1.)
        pe = pe * mask  # zero out positions where mask is 0

        return x + pe

    @property
    def name(self):
        return self._name


class RotaryPositionalEncoding(keras.layers.Layer):
    """
    Rotary Positional Encoding layer for transformer models.
    Applies rotary embeddings to the last two dimensions of the input.
    Args:
        embed_dim (int): Embedding dimension (must be even).
        max_len (int): Maximum sequence length.
    """

    def __init__(self, embed_dim, max_len: int = 100, mask_token: float = -1., pad_token: float = -2., name: str = 'rotary_positional_encoding'):
        super().__init__(name=name)
        assert embed_dim % 2 == 0, "embed_dim must be even for rotary encoding"
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.mask_token = mask_token
        self._name = name
        self.pad_token = pad_token

    def build(self, x):
        # Precompute rotary frequencies
        pos = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]  # (max_len, 1)
        dim = tf.range(self.embed_dim // 2, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim//2)
        inv_freq = 1.0 / (10000 ** (dim / (self.embed_dim // 2)))
        freqs = pos * inv_freq  # (max_len, embed_dim//2)
        self.cos_cached = tf.cast(tf.cos(freqs), tf.float32)  # (max_len, embed_dim//2)
        self.sin_cached = tf.cast(tf.sin(freqs), tf.float32)  # (max_len, embed_dim//2)

    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B, N)
        Returns:
            Tensor with rotary positional encoding applied.
        """
        seq_len = tf.shape(x)[1]
        cos = self.cos_cached[:seq_len, :]  # (N, D//2)
        sin = self.sin_cached[:seq_len, :]  # (N, D//2)
        cos = tf.expand_dims(cos, 0)  # (1, N, D//2)
        sin = tf.expand_dims(sin, 0)  # (1, N, D//2)

        x1, x2 = tf.split(x, 2, axis=-1)  # (B, N, D//2), (B, N, D//2)
        x_rot = tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)  # (B, N, D)

        mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)  # (B, N, 1)
        mask = tf.where(mask == self.pad_token, 0., 1.)
        x_rot = x_rot * mask  # zero out positions where mask is 0

        return x_rot

    @property
    def name(self):
        return self._name


@tf.function
def select_indices(ind, n, m_range):
    """
    Select top-n indices from `ind` (descending sorted) such that:
    - First index is always selected.
    - Each subsequent index has a distance from all previously selected
      indices between m_range[0] and m_range[1], inclusive.
    Args:
        ind: Tensor of shape (B, N) with descending sorted indices.
        n: Number of indices to select.
        m_range: List or tuple [min_distance, max_distance]
    Returns:
        Tensor of shape (B, n) with selected indices per batch.
    """
    m_min = tf.constant(m_range[0], dtype=tf.int32)
    m_max = tf.constant(m_range[1], dtype=tf.int32)

    def per_batch_select(indices):
        top = indices[0]
        selected = tf.TensorArray(dtype=tf.int32, size=n)
        selected = selected.write(0, top)
        count = tf.constant(1)
        i = tf.constant(1)

        def cond(i, count, selected):
            return tf.logical_and(i < tf.shape(indices)[0], count < n)

        def body(i, count, selected):
            candidate = indices[i]
            selected_vals = selected.stack()[:count]
            distances = tf.abs(selected_vals - candidate)
            if_valid = tf.reduce_all(
                tf.logical_and(distances >= m_min, distances <= m_max)
            )
            selected = tf.cond(if_valid,
                               lambda: selected.write(count, candidate),
                               lambda: selected)
            count = tf.cond(if_valid, lambda: count + 1, lambda: count)
            return i + 1, count, selected

        _, _, selected = tf.while_loop(
            cond, body, [i, count, selected],
            shape_invariants=[i.get_shape(), count.get_shape(), tf.TensorShape(None)]
        )
        return selected.stack()

    return tf.map_fn(per_batch_select, ind, dtype=tf.int32)


class AnchorPositionExtractor(keras.layers.Layer):
    @property
    def name(self):
        return self._name

    def __init__(self, num_anchors, dist_thr, name='anchor_extractor', project=True,
                 mask_token=-1., pad_token=-2., return_att_weights=False):
        super().__init__()
        assert isinstance(dist_thr, list) and len(dist_thr) == 2
        assert num_anchors > 0
        self.num_anchors = num_anchors
        self.dist_thr = dist_thr
        self.name = name
        self.project = project
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.return_att_weights = return_att_weights

    def build(self, input_shape):  # att_out (B,N,E)
        b, n, e = input_shape[0], input_shape[1], input_shape[2]
        self.barcode = tf.random.uniform(shape=(1, 1, e))  # add as a token to input
        self.q = self.add_weight(shape=(e, e),
                                 initializer='random_normal',
                                 trainable=True, name=f'query_{self.name}')
        self.k = self.add_weight(shape=(e, e),
                                 initializer='random_normal',
                                 trainable=True, name=f'key_{self.name}')
        self.v = self.add_weight(shape=(e, e),
                                 initializer='random_normal',
                                 trainable=True, name=f'value_{self.name}')
        self.ln = layers.LayerNormalization(name=f'ln_{self.name}')
        if self.project:
            self.g = self.add_weight(shape=(self.num_anchors, e, e),
                                     initializer='random_uniform',
                                     trainable=True, name=f'gate_{self.name}')
            self.w = self.add_weight(shape=(1, self.num_anchors, e, e),
                                     initializer='random_normal',
                                     trainable=True, name=f'w_{self.name}')

    def call(self, input, mask):  # (B,N,E) this is peptide embedding and (B,N) for mask

        mask = tf.cast(mask, tf.float32)  # (B, N)
        mask = tf.where(mask == self.pad_token, 0., 1.)

        barcode = self.barcode
        barcode = tf.broadcast_to(barcode, (tf.shape(input)[0], 1, tf.shape(input)[-1]))  # (B,N,E)
        q = tf.matmul(barcode, self.q)  # (B,1,E)*(E,E)->(B,1,E)
        k = tf.matmul(input, self.k)  # (B,N,E)*(E,E)->(B,N,E)
        v = tf.matmul(input, self.v)  # (B,N,E)*(E,E)->(B,N,E)
        scale = 1 / tf.math.sqrt(tf.cast(tf.shape(input)[-1], tf.float32))
        barcode_att = tf.matmul(q, k, transpose_b=True) * scale  # (B,1,E)*(B,E,N)->(B,1,N)
        # mask: (B,N) => (B,1,N)
        mask_exp = tf.expand_dims(mask, axis=1)
        additive_mask = (1.0 - mask_exp) * -1e9
        barcode_att += additive_mask
        barcode_att = tf.nn.softmax(barcode_att)
        barcode_att *= mask_exp  # to remove the impact of row wise attention of padded tokens. since all are 1e-9
        barcode_out = tf.matmul(barcode_att, v)  # (B,1,N)*(B,N,E)->(B,1,E)
        # barcode_out represents a vector for all information from peptide
        # barcode_att represents the anchor positions which are the tokens with highest weights
        inds, weights, outs = self.find_anchor(input,
                                               barcode_att)  # (B,num_anchors) (B,num_anchors) (B, num_anchors, E)
        if self.project:
            pos_encoding = tf.broadcast_to(
                tf.expand_dims(inds, axis=-1),
                (tf.shape(outs)[0], tf.shape(outs)[1], tf.shape(outs)[2])
            )
            pos_encoding = tf.cast(pos_encoding, tf.float32)
            dim = tf.cast(tf.shape(outs)[-1], tf.float32)
            ra = tf.range(dim, dtype=tf.float32) / dim
            pos_encoding = tf.sin(pos_encoding / tf.pow(40., ra))
            outs += pos_encoding

            weights_bc = tf.expand_dims(weights, axis=-1)
            weights_bc = tf.broadcast_to(weights_bc, (tf.shape(weights_bc)[0],
                                                      tf.shape(weights_bc)[1],
                                                      tf.shape(outs)[-1]
                                                      ))  # (B,num_anchors, E)
            outs = tf.expand_dims(outs, axis=-2)  # (B, num_anchors, 1, E)
            outs_w = tf.matmul(outs, self.w)  # (B,num_anchors,1,E)*(1,num_anchors,E,E)->(B,num_anchors,1,E)
            outs_g = tf.nn.sigmoid(tf.matmul(outs, self.g))
            outs_w = tf.squeeze(outs_w, axis=-2)  # (B,num_anchors,E)
            outs_g = tf.squeeze(outs_g, axis=-2)
            # multiply by attention weights from barcode_att to choose best anchors and additional feature gating
            outs = outs_w * outs_g * weights_bc  # (B, num_anchors, E)
        outs = self.ln(outs)
        # outs -> anchor info, inds -> anchor indeces, weights -> anchor att weights, barcode_out -> whole peptide features
        # (B,num_anchors,E), (B,num_anchors), (B,num_anchors), (B,E)
        if self.return_att_weights:
            return outs, inds, weights, tf.squeeze(barcode_out, axis=1), barcode_att
        else:
            return outs, inds, weights, tf.squeeze(barcode_out, axis=1)

    def find_anchor(self, input, barcode_att):  # (B,N,E), (B,1,N)
        inds = tf.argsort(barcode_att, axis=-1, direction='DESCENDING', stable=False)  # (B,1,N)
        inds = tf.squeeze(inds, axis=1)  # (B,N)
        selected_inds = select_indices(inds, n=self.num_anchors, m_range=self.dist_thr)  # (B,num_anchors)
        sorted_selected_inds = tf.sort(selected_inds)
        sorted_selected_weights = tf.gather(tf.squeeze(barcode_att, axis=1),
                                            sorted_selected_inds,
                                            axis=1,
                                            batch_dims=1)  # (B,num_anchors)
        sorted_selected_output = tf.gather(input, sorted_selected_inds, axis=1, batch_dims=1)  # (B,num_anchors,E)
        return sorted_selected_inds, sorted_selected_weights, sorted_selected_output

    @name.setter
    def name(self, value):
        self._name = value


def generate_mhc(samples=1024, min_len=5, max_len=15, dim=16):
    X_list = []
    mask_list = []

    for _ in range(samples):
        l = np.random.randint(min_len, max_len + 1)
        x = np.random.rand(l, dim).astype(np.float32)
        X_list.append(x)
        mask = np.ones((l,), dtype=bool)
        mask_list.append(mask)

    # Pad sequences
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X_list, maxlen=max_len, dtype='float32', padding='post'
    )  # shape: (samples, max_len, dim)

    mask_padded = tf.keras.preprocessing.sequence.pad_sequences(
        mask_list, maxlen=max_len, dtype=bool, padding='post'
    )  # shape: (samples, max_len)

    # Compute masked mean (only over valid tokens)
    mask_exp = np.expand_dims(mask_padded, axis=-1).astype(np.float32)  # (samples, max_len, 1)
    y = np.sum(X_padded * mask_exp, axis=1) / np.maximum(np.sum(mask_exp, axis=1), 1e-8)  # (samples, dim)
    mean_of_all = tf.reduce_mean(y)
    y = tf.reduce_mean(y, axis=-1, keepdims=True)
    y = np.where(y > mean_of_all, 1., 0.)
    mask_padded = np.where(mask_padded == False, -2., 0.)

    return X_padded, y, mask_padded

def generate_peptide(samples=1024, min_len=5, max_len=15, k=9):
    """
    Generate random peptide one-hot tensors of shape (N, RF, k, 21)
    where RF = max_len - k + 1.
    """
    peptides = []
    for _ in range(samples):
        l = np.random.randint(min_len, max_len + 1)
        seq = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), l)
        seq_str = ''.join(seq)
        peptides.append(seq_str)
    # Convert to tf.Tensor
    seqs = tf.constant(peptides)
    RF = max_len - k + 1
    onehot = peptides_to_onehot_tf(seqs, max_len, k)  # (N, RF, k, 21)
    return onehot.numpy()



# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
MASK_TOKEN = -1
PAD_TOKEN  = -2.

def make_rf_mask(pep_batch: tf.Tensor) -> tf.Tensor:
    """Return (B, RF) float mask for a peptide batch of shape *(B, RF,K,21)*.

    A slot is **1** when any (K,21) element is non‑zero, else **PAD_TOKEN**.
    """
    # True where *any* channel is non‑zero → valid RF window
    non_zero = tf.math.reduce_any(tf.not_equal(pep_batch, 0.), axis=[2, 3])     # (B, RF)
    return tf.where(non_zero, 1., PAD_TOKEN)                                    # float32


#### define model
# input layers
def build_custom_classifier(max_len_peptide: int,
                            max_len_mhc: int,
                            k: int = 9,
                            embed_dim_pep: int = 64,
                            embed_dim_mhc: int = 128,
                            mask_token: float = MASK_TOKEN,
                            pad_token: float = PAD_TOKEN):
    """Return a compiled Keras model that consumes

        pep_input : (RF_max, k, 21)
        mhc_input : (max_len_mhc, 1152)

    RF_max is ``max_len_peptide − k + 1``.
    """
    RF_max = max_len_peptide - k + 1

    # ----- inputs ----------------------------------------------------------
    pep_input = keras.Input(shape=(RF_max, k, 21),       name="pep_input")
    mhc_input = keras.Input(shape=(max_len_mhc, 1152),   name="mhc_input")

    # ----- peptide branch --------------------------------------------------
    pep_mask = layers.Lambda(make_rf_mask, name="pep_mask")(pep_input)         # (B, RF)

    pep_flat = layers.Reshape((RF_max, k * 21), name="pep_flat")(pep_input)    # (B, RF, k·21)

    # TODO, check if it make sence to do this before or after positional encoding
    pep_proj = layers.Dense(embed_dim_pep, activation="relu",
                            name="pep_proj1")(pep_flat)  # (B, RF, 64)

    pep_pe = RotaryPositionalEncoding(embed_dim_pep, max_len=RF_max,
                                mask_token=mask_token, pad_token=pad_token,
                                name="pep_pos_enc")(pep_proj, pep_mask)

    pep_att1 = AttentionLayer(input_dim=embed_dim_pep, output_dim=embed_dim_pep,
                              type="self", heads=4, name="pep_self_att1",
                              mask_token=mask_token, pad_token=pad_token)(
                    pep_pe, mask=pep_mask)

    # ----- MHC branch ------------------------------------------------------
    mhc_mask_bool = tf.math.reduce_any(tf.not_equal(mhc_input, 0.), axis=-1)
    mhc_mask      = layers.Lambda(lambda x: tf.where(x, 1., pad_token),
                                  name="mhc_mask")(mhc_mask_bool)

    # TODO check if it is correct
    mhc_proj1 = layers.Dense(embed_dim_mhc, activation="relu",
                             name="mhc_proj1")(mhc_input)  # (B, L_mhc, D_mhc)

    mhc_pe = RotaryPositionalEncoding(embed_dim=embed_dim_mhc,max_len=max_len_mhc,
                                        mask_token=mask_token,pad_token=pad_token,
                                        name="mhc_pos_enc")(mhc_proj1, mhc_mask)

    mhc_att1  = AttentionLayer(input_dim=embed_dim_mhc, output_dim=embed_dim_mhc,
                               type="self", heads=4, name="mhc_self_att1",
                               mask_token=mask_token, pad_token=pad_token)(
                    mhc_pe, mask=mhc_mask)
    mhc_att2, att_w1 = AttentionLayer(input_dim=embed_dim_mhc, output_dim=embed_dim_mhc,
                                 type="self", heads=4, name="mhc_self_att2",
                                 resnet=False, mask_token=mask_token, pad_token=pad_token,
                                 return_att_weights=True)(
                      mhc_att1, mask=mhc_mask)                                  # (B, L_mhc, D_mhc), att_weights


    # NEW: project MHC tokens → pep embed‑dim (64) so Q/K/V dims align
    mhc_to_D_pep = layers.Dense(embed_dim_pep, activation="relu",
                              name="mhc_to_D_pep")(mhc_att2)                    # (B, L_mhc, D_pep)

    # ----- cross‑attention -------------------------------------------------
    cross_att = AttentionLayer(input_dim=embed_dim_pep, output_dim=embed_dim_pep,
                           type="cross", heads=8, name="cross_att",
                           mask_token=mask_token, pad_token=pad_token)(
                 pep_att1, mask=pep_mask,
                 context=mhc_to_D_pep, context_mask=mhc_mask)

    cross_proj = layers.Dense(128, activation="relu", name="cross_proj")(cross_att)

    final_att = AttentionLayer(input_dim=128, output_dim=128, type="self",
                               heads=2, name="final_pep_self_att",
                               return_att_weights=True,
                               mask_token=mask_token, pad_token=pad_token)(
                    cross_proj, mask=pep_mask)

    final_features = AnchorPositionExtractor(num_anchors=2, dist_thr=[8, 15], # outs, inds, weights, barcode_out, barcode_att
                                      name="anchor_extractor", # (B,num_anchors,E), (B,num_anchors), (B,num_anchors), (B,E), (B,N,N)
                                      project=True,
                                      return_att_weights=True,
                                      mask_token=mask_token, pad_token=pad_token)(
                   final_att[0], mask=pep_mask)

    # ----------------------------------------------------------------------
    #  Three heads (barcode, anchors, pooled) — unchanged
    # ----------------------------------------------------------------------
    # barcode_vec = final_features[-2]
    # x_bc = layers.Dense(32, activation="relu", name="barcodout1_dense")(barcode_vec)
    # x_bc = layers.Dropout(0.3, name="barcodout1_dropout")(x_bc)
    # out_bc = layers.Dense(1, activation="sigmoid", name="barcode_cls")(x_bc)

    anchor_feat = final_features[0]
    x_an = layers.Flatten(name="anchorout_flatten")(anchor_feat)
    x_an = layers.Dense(64, activation="relu", name="anchorout1_dense")(x_an)
    x_an = layers.Dropout(0.3, name="anchorout1_dropout")(x_an)
    x_an = layers.Dense(16, activation="relu", name="anchorout2_dense")(x_an)
    x_an = layers.Dropout(0.3, name="anchorout2_dropout")(x_an)
    out_an = layers.Dense(1, activation="sigmoid", name="anchor_cls")(x_an)

    # pooled = layers.GlobalAveragePooling1D(name="attout_gap")(final_att[0])
    # x_po = layers.Dense(32, activation="relu", name="attout1_dense")(pooled)
    # x_po = layers.Dropout(0.3, name="attout1_dropout")(x_po)
    # out_po = layers.Dense(1, activation="sigmoid", name="attn_cls")(x_po)

    model = keras.Model(inputs=[pep_input, mhc_input],
                        outputs=out_an,
                        name="PeptideMHC_CrossAtt")

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics={
            "anchor_cls":  ["binary_accuracy", "AUC"],
        },
    )
    return model

# def main():
#     max_len_peptide = 15
#     k = 9
#     max_len_mhc = 40
#     RF_max = max_len_peptide - k + 1
#
#     model = build_custom_classifier(max_len_peptide, max_len_mhc, k=k)
#     model.summary(line_length=110)
#
#     batch = 16
#     # pep_dummy = np.zeros((batch, RF_max, k, 21), dtype=np.float32)
#     # pep_dummy[:, :3] = np.random.rand(batch, 3, k, 21)
#     #
#     # mhc_dummy = np.zeros((batch, max_len_mhc, 1152), dtype=np.float32)
#     # mhc_dummy[:, :25] = np.random.rand(batch, 25, 1152)
#
#     pep_syn = generate_peptide(samples=1000, min_len=9, max_len=max_len_peptide, k=k)
#     mhc_syn, _, mhc_mask = generate_mhc(samples=1000, min_len=25, max_len=max_len_mhc, dim=1152)
#
#     y = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)
#
#     history = model.fit(x=[pep_syn, mhc_syn], y=y, epochs=10, batch_size=batch)
#
#     # save model
#     model_dir = pathlib.Path("model_output")
#     model_dir.mkdir(parents=True, exist_ok=True)
#     model_path = model_dir / "peptide_mhc_cross_attention_model.h5"
#     model.save(model_path)
#     print(f"Model saved to {model_path}")
#
#     print("Sanity‑check complete — no dimension errors.")
#
#     # PREDICT
#     preds = model.predict([pep_syn[:batch], mhc_syn[:batch]])
#     print("Predictions for first batch:")
#     for i, pred in enumerate(preds):
#         print(f"Sample {i + 1}: Anchor: {pred[0]:.4f}")
#     # Save model metadata
#     metadata = {
#         "max_len_peptide": max_len_peptide,
#         "k": k,
#         "max_len_mhc": max_len_mhc,
#         "RF_max": RF_max,
#         "embed_dim_pep": 64,
#         "embed_dim_mhc": 128,
#         "mask_token": MASK_TOKEN,
#         "pad_token": PAD_TOKEN
#     }
#     metadata_path = model_dir / "model_metadata.json"
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=4)
#
#     print(f"Model metadata saved to {metadata_path}")
#
#     # plot metrics and confusion
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import pandas as pd
#     history_df = pd.DataFrame(history.history)
#     print(f"Keys in history: {list(history_df.columns)}")
#
#     ## Plot metrics with error handling and saving to disk
#     metrics_dir = model_dir / "metrics"
#     metrics_dir.mkdir(parents=True, exist_ok=True)
#
#     # Plot binary accuracy
#     plt.figure(figsize=(10, 5))
#     sns.lineplot(data=history_df, x=history_df.index, y='binary_accuracy', label='Binary Accuracy')
#     plt.title('Binary Accuracy Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Binary Accuracy')
#     plt.legend()
#     plt.show()
#
#     # plot AUC
#     plt.figure(figsize=(10, 5))
#     sns.lineplot(data=history_df, x=history_df.index, y='auc', label='AUC')
#     plt.title('AUC Over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('AUC')
#     plt.legend()
#     plt.show()
# if __name__ == "__main__":
#     main()
