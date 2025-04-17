import tensorflow as tf
import keras
from keras import layers


# ------------------------------
# SCQ Layer as defined before
# ------------------------------
class SCQ(layers.Layer):
    def __init__(self, num_embed=64, dim_embed=32, dim_input=128, lambda_reg=0.1, proj_iter=10,
                 descrete_loss=False, beta_loss=0.25, **kwargs):
        """
        Soft Convex Quantization (SCQ) layer.

        Args:
            num_embed: Number of codebook vectors.
            dim_embed: Embedding dimension (for both encoder and codebook).
            dim_input: Dimension of input features (projected to dim_embed).
            lambda_reg: Regularization parameter in the SCQ objective.
            proj_iter: Number of iterations for the iterative simplex projection.
            descrete_loss: if True, commitment and codebook loss are calculated similar
                to VQ-VAE loss, with stop-gradient operation. If False, only quantization error
                is calculated as loss |Z_q - Z_e|**2
            beta_loss: A float to be used in VQ loss. Only used if descrete_loss==True.
                multiplied to (1-b)*commitment_loss + b*codebook_loss
        """
        super().__init__(**kwargs)
        self.num_embed = num_embed
        self.dim_embed = dim_embed
        self.dim_input = dim_input
        self.lambda_reg = lambda_reg
        self.proj_iter = proj_iter
        self.descrete_loss = descrete_loss
        self.beta_loss = beta_loss
        self.epsilon = 1e-5

    def build(self, input_shape):
        # Learnable scale and bias for layer normalization.
        self.gamma = self.add_weight(
            shape=(self.dim_embed,),
            initializer="ones",
            trainable=True,
            name="ln_gamma")
        self.beta = self.add_weight(
            shape=(self.dim_embed,),
            initializer="zeros",
            trainable=True,
            name="ln_beta")
        # Codebook: each column is one codebook vector.
        self.embed_w = self.add_weight(
            shape=(self.dim_embed, self.num_embed),
            initializer='uniform',
            trainable=True,
            name='scq_embedding')
        # Projection weights to map input features into the embedding space.
        self.d_w = self.add_weight(
            shape=(self.dim_input, self.dim_embed),
            initializer='random_normal',
            trainable=True,
            name='scq_w')
        self.d_b = self.add_weight(
            shape=(self.dim_embed,),
            initializer='zeros',
            trainable=True,
            name='scq_b')

    def call(self, inputs):
        """
        Forward pass for SCQ.
        Args:
            inputs: Tensor of shape (B, N, dim_input)
        Returns:
            Quantized output: Tensor of shape (B, N, dim_embed)
        """
        # 1. Project inputs to the embedding space and apply layer normalization.
        x = tf.matmul(inputs, self.d_w) + self.d_b  # (B, N, dim_embed)
        x = self.layernorm(x)

        input_shape = tf.shape(x)  # (B, N, dim_embed)
        flat_inputs = tf.reshape(x, [-1, self.dim_embed])  # (B*N, dim_embed)

        # 2. Compute hard VQ assignments as initialization.
        flat_detached = tf.stop_gradient(flat_inputs)
        x_norm_sq = tf.reduce_sum(flat_detached ** 2, axis=1, keepdims=True)  # (B*N, 1)
        codebook_norm_sq = tf.reduce_sum(self.embed_w ** 2, axis=0, keepdims=True)  # (1, num_embed)
        dot_product = tf.matmul(flat_detached, self.embed_w)  # (B*N, num_embed)
        distances = x_norm_sq + codebook_norm_sq - 2 * dot_product  # (B*N, num_embed)

        assign_indices = tf.argmin(distances, axis=1)  # (B*N,)
        P_tilde = tf.one_hot(assign_indices, depth=self.num_embed, dtype=tf.float32)  # (B*N, num_embed)
        P_tilde = tf.transpose(P_tilde)  # (num_embed, B*N)

        # 3. Solve the SCQ optimization via a linear system.
        C = self.embed_w  # (dim_embed, num_embed)
        Z = tf.transpose(flat_inputs)  # (dim_embed, B*N)
        CtC = tf.matmul(C, C, transpose_a=True)  # (num_embed, num_embed)
        I = tf.eye(self.num_embed, dtype=tf.float32)
        A = CtC + self.lambda_reg * I  # (num_embed, num_embed)
        CtZ = tf.matmul(C, Z, transpose_a=True)  # (num_embed, B*N)
        b = CtZ + self.lambda_reg * P_tilde  # (num_embed, B*N)
        P_sol = tf.linalg.solve(A, b)  # Unconstrained solution: (num_embed, B*N)

        # 4. Project each column of P_sol onto the probability simplex.
        P_proj = self.project_columns_to_simplex(P_sol)  # (num_embed, B*N)
        # P_proj = tf.nn.softmax(P_sol, axis=0)
        out_P_proj = tf.transpose(P_proj)  # (B*N, num_embed)
        out_P_proj = tf.reshape(out_P_proj, (input_shape[0], input_shape[1], -1))  # (B,N,num_embed)
        # out_P_proj = tf.reduce_mean(out_P_proj, axis=-1, keepdims=True) #(B,N,1)

        # 5. Reconstruct quantized output: Z_q = C * P_proj.
        Zq_flat = tf.matmul(C, P_proj)  # (dim_embed, B*N)
        Zq = tf.transpose(Zq_flat)  # (B*N, dim_embed)

        # 6. calculate quantization loss, combined commitment and codebook losses
        if not self.descrete_loss:
            loss = tf.reduce_mean((Zq - flat_inputs) ** 2)  # (B*N, dim_embed)
        else:
            commitment_loss = tf.reduce_mean((tf.stop_gradient(Zq) - flat_inputs) ** 2)
            codebook_loss = tf.reduce_mean((Zq - flat_detached) ** 2)
            loss = (
                    (tf.cast(1 - self.beta_loss, tf.float32) * commitment_loss) +
                    (tf.cast(self.beta, tf.float32) * codebook_loss)
            )

        Zq = tf.reshape(Zq, input_shape)  # (B, N, dim_embed)
        return Zq, out_P_proj, loss  # (B,N,embed_dim), #(B,N,num_embed), (1,)

    def project_columns_to_simplex(self, P_sol):
        """Projects columns of a matrix to the probability simplex."""
        # Transpose to work with rows instead of columns
        P_t = tf.transpose(P_sol)  # (B*N, num_embed)
        # Sort rows in descending order
        sorted_P = tf.sort(P_t, axis=1, direction='DESCENDING')
        # Compute cumulative sums and thresholds
        cumsum = tf.cumsum(sorted_P, axis=1)
        k = tf.range(1, tf.shape(sorted_P)[1] + 1, dtype=tf.float32)
        thresholds = (cumsum - 1.0) / k
        # Find maximum valid index per row (cast to int32 explicitly)
        valid_mask = sorted_P > thresholds
        max_indices = tf.argmax(
            tf.cast(valid_mask, tf.int32) *
            tf.range(tf.shape(sorted_P)[1], dtype=tf.int32),
            axis=1
        )
        max_indices = tf.cast(max_indices, tf.int32)  # Explicit cast to int32
        # Gather required cumulative sums
        batch_indices = tf.range(tf.shape(P_t)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, max_indices], axis=1)
        # Rest of the code remains the same...
        selected_cumsum = tf.gather_nd(cumsum, gather_indices)
        theta = (selected_cumsum - 1.0) / tf.cast(max_indices + 1, tf.float32)
        # Apply projection and transpose back
        P_projected = tf.nn.relu(P_t - theta[:, tf.newaxis])
        return tf.transpose(P_projected)

    def layernorm(self, inputs):
        """
        Custom layer normalization over the last dimension.
        """
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.math.reduce_variance(inputs, axis=-1, keepdims=True)
        normed = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normed + self.beta


class Encoder(layers.Layer):
    def __init__(self, dim_input, dim_embed, heads=4, names='encoder', **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self.dim_embed = dim_embed
        self.heads = heads
        self.names = names
        self.epsilon = 1e-5
        self.scale = 1 / tf.sqrt(tf.cast(dim_embed, dtype=tf.float32))

    def build(self, input_shape):
        # Learnable scale and bias for layer normalization.
        self.gamma = self.add_weight(
            shape=(self.dim_embed,),
            initializer="ones",
            trainable=True,
            name=f"ln_gamma_{self.names}")
        self.beta = self.add_weight(
            shape=(self.dim_embed,),
            initializer="zeros",
            trainable=True,
            name=f"ln_beta_{self.names}")
        # Projection weights to map input features into the embedding space.
        self.d_w = self.add_weight(
            shape=(self.dim_input, self.dim_embed),
            initializer='random_normal',
            trainable=True,
            name=f'd_w_{self.names}')
        self.d_b = self.add_weight(
            shape=(self.dim_embed,),
            initializer='zeros',
            trainable=True,
            name=f'd_b_{self.names}')
        # gating
        self.d_g = self.add_weight(
            shape=(self.heads, 1, self.dim_embed, self.dim_embed),
            initializer='uniform',
            trainable=True,
            name=f'g_w_{self.names}')
        # attention
        self.q = self.add_weight(
            shape=(self.heads, 1, self.dim_embed, self.dim_embed),
            initializer='random_normal',
            trainable=True,
            name=f'query_{self.names}')
        self.k = self.add_weight(
            shape=(self.heads, 1, self.dim_embed, self.dim_embed),
            initializer='random_normal',
            trainable=True,
            name=f'key_{self.names}')
        self.v = self.add_weight(
            shape=(self.heads, 1, self.dim_embed, self.dim_embed),
            initializer='random_normal',
            trainable=True,
            name=f'value_{self.names}')
        # final projection
        self.d_o = self.add_weight(shape=(int(self.heads * self.dim_embed), self.dim_embed),
                                   initializer='random_normal',
                                   trainable=True, name=f'dout_{self.names}')


    def call(self, inputs):
        x = tf.matmul(inputs, self.d_w) + self.d_b  # (B,N,I)->(B,N,E)
        x = self.layernorm(x)
        # attention
        query = tf.matmul(x, self.q)  # (B,N,E)@(H,1,E,E)->(H,B,N,E)
        key = tf.matmul(x, self.k)
        value = tf.matmul(x, self.v)
        gate = tf.nn.sigmoid(tf.matmul(x, self.d_g))

        qk = tf.einsum('hbni,hbki->hbnk', query, key)  # (H,B,N,N)
        qk *= self.scale
        att = tf.nn.softmax(qk)
        out = tf.matmul(att, value)  # (H,B,N,N)@(H,B,N,E)->(H,B,N,E)
        out = tf.multiply(out, gate)  # gate

        out = tf.transpose(out, perm=[1, 2, 3, 0])
        b, h, n, o = tf.shape(out)[0], tf.shape(out)[-1], tf.shape(out)[1], tf.shape(out)[2]
        out = tf.reshape(out, [b, n, h * o])  # (B, N, H * E)
        out = tf.matmul(out, self.d_o)  # (B,N,H*E)@(H*E,E)->(B,N,E)

        return out

    def layernorm(self, inputs):
        """
        Custom layer normalization over the last dimension.
        """
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.math.reduce_variance(inputs, axis=-1, keepdims=True)
        normed = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normed + self.beta


class Decoder(layers.Layer):
    def __init__(self, dim_input, dim_embed, dim_output, heads=4, names='decoder', **kwargs):
        super().__init__()
        self.dim_input = dim_input
        self.dim_embed = dim_embed
        self.dim_output = dim_output
        self.heads = heads
        self.names = names
        self.epsilon = 1e-5
        self.scale = 1 / tf.sqrt(tf.cast(dim_embed, dtype=tf.float32))

    def build(self, input_shape):
        self.w_in = self.add_weight(shape=(self.dim_input, self.dim_embed), trainable=True, initializer='random_normal',
                                    name=f'w_in_{self.names}')
        self.b_in = self.add_weight(shape=(self.dim_embed,), trainable=True, initializer='zeros',
                                    name=f'b_in_{self.names}')
        # Learnable scale and bias for layer normalization.
        self.gamma = self.add_weight(shape=(self.dim_embed,), initializer="ones", trainable=True,
                                     name=f"ln_gamma_{self.names}")
        self.beta = self.add_weight(shape=(self.dim_embed,), initializer="zeros", trainable=True,
                                    name=f"ln_beta_{self.names}")
        # attetnion
        self.q = self.add_weight(shape=(self.heads, 1, self.dim_embed, self.dim_embed), trainable=True,
                                 initializer='random_normal', name=f'query_{self.names}')
        self.k = self.add_weight(shape=(self.heads, 1, self.dim_embed, self.dim_embed), trainable=True,
                                 initializer='random_normal', name=f'key_{self.names}')
        self.v = self.add_weight(shape=(self.heads, 1, self.dim_embed, self.dim_embed), trainable=True,
                                 initializer='random_normal', name=f'value_{self.names}')
        # project out
        self.w_out = self.add_weight(shape=(int(self.heads * self.dim_embed), self.dim_output), trainable=True,
                                     initializer='random_normal', name=f'w_out_{self.names}')
        self.b_out = self.add_weight(shape=(self.dim_output,), trainable=True, initializer='zeros',
                                     name=f'b_out_{self.names}')

        # mlp head
        self.mlp_head1 = layers.Dense(self.dim_output, activation='tanh', name=f'mlp_head_{self.names}')

    def call(self, inputs):
        Zq = inputs  # (B,N,embed_dim), #(B,N,1), (1,)
        x = tf.matmul(Zq, self.w_in) + self.b_in
        x = self.layernorm(x)  # (B,N,dim_embed)
        # attention
        query = tf.matmul(x, self.q)  # (B,N,dim_emned)@(H,1,dim_embed,dim_embed)->(H,B,N,dim_embed)
        key = tf.matmul(x, self.k)
        value = tf.matmul(x, self.v)
        qk = tf.einsum('hbni,hbji->hbnj', query, key)  # (H,B,N,N)
        qk *= self.scale
        att = tf.nn.softmax(qk)
        out = tf.matmul(att, value)  # (H,B,N,N)@(H,B,N,dim_embed)
        # out projection
        out = tf.transpose(out, perm=[1, 2, 3, 0])
        b, h, n, o = tf.shape(out)[0], tf.shape(out)[-1], tf.shape(out)[1], tf.shape(out)[2]
        out = tf.reshape(out, [b, n, h * o])  # (B, N, H * E)
        out = tf.matmul(out, self.w_out) + self.b_out  # (B,N,H*E)@(H*E,O)->(B,N,O)
        # out = tf.nn.softmax(out) # Disable softmax for regression tasks
        # out = tf.nn.sigmoid(out)
        # mlp head
        out = self.mlp_head1(out)
        return out

    def layernorm(self, inputs):
        """
        Custom layer normalization over the last dimension.
        """
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.math.reduce_variance(inputs, axis=-1, keepdims=True)
        normed = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normed + self.beta


class SCQ_model(tf.keras.models.Model):
    def __init__(self, input_dim=1024,  general_embed_dim=128, codebook_dim=16, codebook_num=64,
                 descrete_loss=False, heads=4, names='SCQ_model',
                 weight_recon=1, weight_vq=0.2, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.general_embed_dim = general_embed_dim
        self.codebook_dim = codebook_dim
        self.codebook_num = codebook_num
        self.descrete_loss = descrete_loss
        self.heads = heads
        self.names = names
        self.weight_recon = tf.cast(weight_recon, tf.float32)
        self.weight_vq = tf.cast(weight_vq, tf.float32)
        # define model
        # self.mlp_input = layers.Dense(self.input_dim, activation='swish', name=f'mlp_input_{self.names}')
        self.encoder = Encoder(dim_input=self.input_dim, dim_embed=self.general_embed_dim, heads=self.heads)

        self.scq = SCQ(dim_input=self.general_embed_dim, dim_embed=self.codebook_dim,
                       num_embed=self.codebook_num, descrete_loss=self.descrete_loss)

        self.decoder = Decoder(dim_input=self.codebook_dim, dim_embed=self.general_embed_dim,
                               dim_output=self.input_dim, heads=self.heads)
        # self.mlp_output = layers.Dense(self.input_dim, activation='swish', name=f'mlp_output_{self.names}')

        # Loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.vq_loss_tracker = tf.keras.metrics.Mean(name='vq_loss')
        self.perplexity_tracker = tf.keras.metrics.Mean(name='perplexity')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker,
            self.perplexity_tracker
        ]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Forward pass
            # x = self.mlp_input(x)
            encoded = self.encoder(x)
            Zq, out_P_proj, vq_loss = self.scq(encoded)  # (B,N,E),(B,N,K),(1,)
            decoded = self.decoder(Zq)
            # out_P_proj = self.mlp_output(out_P_proj)  # (B,N,K)

            # Calculate losses using the new method
            losses = self.compute_losses(x, decoded, vq_loss, out_P_proj)

        # Update weights
        vars = self.trainable_weights
        grads = tape.gradient(losses['total_loss'], vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        # Update metrics
        self.total_loss_tracker.update_state(losses['total_loss'])
        self.recon_loss_tracker.update_state(losses['recon_loss'])
        self.vq_loss_tracker.update_state(losses['vq_loss'])
        self.perplexity_tracker.update_state(losses['perplexity'])

        return {
            'loss': self.total_loss_tracker.result(),
            'recon': self.recon_loss_tracker.result(),
            'vq': self.vq_loss_tracker.result(),
            'perplexity': self.perplexity_tracker.result()
        }

    def test_step(self, data):
            # Get inputs from data
            x = data

            # Forward pass
            encoded = self.encoder(x)
            Zq, out_P_proj, vq_loss = self.scq(encoded)
            decoded = self.decoder(Zq)

            # Calculate losses using the new method
            losses = self.compute_losses(x, decoded, vq_loss, out_P_proj)

            # Update metrics
            self.total_loss_tracker.update_state(losses['total_loss'])
            self.recon_loss_tracker.update_state(losses['recon_loss'])
            self.vq_loss_tracker.update_state(losses['vq_loss'])
            self.perplexity_tracker.update_state(losses['perplexity'])

            return {
                'loss': self.total_loss_tracker.result(),
                'recon': self.recon_loss_tracker.result(),
                'vq': self.vq_loss_tracker.result(),
                'perplexity': self.perplexity_tracker.result()
            }



    def call(self, inputs, training=False):
        encoded = self.encoder(inputs)
        Zq, out_P_proj, vq_loss = self.scq(encoded)  # Ignore the loss output
        decoded = self.decoder(Zq)
        return decoded, Zq, out_P_proj  # Return final output

    def compute_perplexity(slef, out_P_proj):
        p_j = tf.reduce_mean(out_P_proj, axis=[0, 1])  # (B, N, K) -> (K,)
        p_j = tf.clip_by_value(p_j, 1e-10, 1 - (1e-9))
        p_j = p_j / tf.reduce_sum(p_j)  # Normalize to ensure sum to 1
        entropy = -tf.reduce_sum(p_j * tf.math.log(p_j) / tf.math.log(2.0))  # Entropy: -sum(p_j * log2(p_j))
        perplexity = tf.pow(2.0, entropy)  # Perplexity: 2^entropy
        return perplexity

    def compute_losses(self, x, decoded, vq_loss, out_P_proj):
        """Compute all losses with improved numerical stability."""
        # Clip to avoid numerical issues
        decoded_clipped = tf.clip_by_value(decoded, 1e-10, 1.0 - 1e-10)

        # Ensure tensors have compatible shapes
        # Get input shapes for debugging
        x_shape = tf.shape(x)
        decoded_shape = tf.shape(decoded_clipped)

        # Reshape if necessary to ensure compatibility
        if len(x.shape) != len(decoded_clipped.shape):
            # Make sure both tensors are 3D (batch, seq_len, features)
            if len(x.shape) == 2:
                x = tf.expand_dims(x, axis=1)
            if len(decoded_clipped.shape) == 2:
                decoded_clipped = tf.expand_dims(decoded_clipped, axis=1)

        # Calculate reconstruction loss - element-wise cross-entropy
        # We'll use the manual formula for more control over dimensions
        epsilon = 1e-10
        # recon_loss = -tf.reduce_sum(x * tf.math.log(decoded_clipped + epsilon), axis=-1)
        # recon_loss = tf.reduce_mean(tf.square(x - decoded))
        # for sigmoid
        # recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, tf.nn.sigmoid(decoded)))
        # normal loss
        recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, decoded_clipped))


        # Compute perplexity (codebook usage metric)
        perplexity = self.compute_perplexity(out_P_proj)

        # Add KL regularization to encourage more uniform codebook usage
        p_j = tf.reduce_mean(out_P_proj, axis=[0, 1])
        print("p_j:", p_j)
        p_uniform = tf.ones_like(p_j) / tf.cast(tf.shape(p_j)[0], tf.float32)
        # Use tf.clip_by_value to prevent log(0) issues
        p_j_safe = tf.clip_by_value(p_j, epsilon, 1.0)
        p_uniform_safe = tf.clip_by_value(p_uniform, epsilon, 1.0)
        kl_loss = tf.reduce_sum(p_j_safe * tf.math.log(p_j_safe / p_uniform_safe))
        # Combine losses with weights
        total_loss = (self.weight_recon * recon_loss +
                      self.weight_vq * vq_loss +
                      0.1 * kl_loss)

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'kl_loss': kl_loss,
            'perplexity': perplexity
        }


'''import tensorflow as tf

import tensorflow as tf


def random_one_hot_sequence(batch, seq_len, num_classes=21, class_probs=None):
    """
    Generates a random one-hot encoded tensor of shape (batch, seq_len, num_classes)
    using a multinomial distribution for more varied sampling.

    Args:
        batch: Number of sequences (batch size).
        seq_len: Length of each sequence.
        num_classes: Number of classes for one-hot encoding (default is 21).
        class_probs: Optional tensor of shape (num_classes,) defining the probability
                     distribution over classes. If None, assumes uniform distribution.

    Returns:
        A tensor of shape (batch, seq_len, num_classes) with one-hot encoded values.
    """
    # Define class probabilities (uniform if not provided)
    if class_probs is None:
        class_probs = tf.ones([num_classes], dtype=tf.float32) / num_classes
    else:
        class_probs = tf.convert_to_tensor([class_probs], dtype=tf.float32)
        class_probs = class_probs / tf.reduce_sum(class_probs)  # Normalize to sum to 1

    # Expand class probabilities to match the number of samples
    logits = tf.math.log(class_probs)[tf.newaxis, :]  # Shape: (1, num_classes)
    logits = tf.tile(logits, [batch * seq_len, 1])  # Shape: (batch * seq_len, num_classes)

    # Sample indices from the multinomial distribution
    random_indices = tf.random.categorical(logits=logits, num_samples=1)  # Shape: (batch * seq_len, 1)
    random_indices = tf.reshape(random_indices, [batch, seq_len])  # Reshape to (batch, seq_len)

    # Apply one-hot encoding
    one_hot_encoded = tf.one_hot(random_indices, depth=num_classes)

    return one_hot_encoded'''


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


import tensorflow as tf
import numpy as np
import pandas as pd
import os

# # Set parameters
# os.makedirs('test_tmp', exist_ok=True)
# batch_size = 600
# seq_length = 20  # Example sequence length
# feature_dim = 21  # Must match encoder input dim
# # Generate random input data (batch_size, seq_length, feature_dim)
# x_train = random_one_hot_sequence(batch_size, seq_length, feature_dim)
# # Initialize SCQ model
# model = SCQ_model(input_dim=feature_dim, general_embed_dim=128, codebook_dim=32, codebook_num=5, descrete_loss=False, heads=8)
# # Compile model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
# # Train model and capture history
# history = model.fit(x_train, epochs=1000, batch_size=batch_size)
# # Convert history to DataFrame and save as CSV
# history_df = pd.DataFrame(history.history)
# history_df.to_csv("test_tmp/model_history.csv", index=False)
# # Test model on new data
# x_test = random_one_hot_sequence(batch_size, seq_length, feature_dim)
# output = model(x_test)
# # Save input and output arrays as .npy files
#
# os.makedirs
# np.save("test_tmp/input_data.npy", x_test)
# np.savez("test_tmp/output_data.npz", decoded=output[0].numpy(), zq=output[1].numpy(), pj=output[2].numpy())
# print("Training complete. Model history saved as 'model_history.csv'.")
# print("Input and output arrays saved as 'input_data.npy' and 'output_data.npy'.")
# print((output[2]))
# print(x_train[0])

# # Set parameters
# os.makedirs('test_tmp', exist_ok=True)
# batch_size = 4
# seq_length = 1024  # this is the length of the latent from pep2vec
# feature_dim = 1  # we have the latent representation
# mhc1_pep2vec_embeddings = pd.read_parquet("../data/Pep2Vec/wrapper_mhc1.parquet")
# # Select the latent columns (the columns that has latent in their name)
# mhc1_latent_columns = [col for col in mhc1_pep2vec_embeddings.columns if 'latent' in col]
# print(f"Found {len(mhc1_latent_columns)} latent columns for MHC1")
# X_mhc1 = mhc1_pep2vec_embeddings[mhc1_latent_columns].values
# X = X_mhc1.reshape(-1, seq_length, feature_dim)
# from sklearn.model_selection import train_test_split
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
#
# # Create datasets
# train_dataset = create_dataset(X_train, batch_size=batch_size, is_training=True)
# test_dataset = create_dataset(X_test, batch_size=batch_size, is_training=False)
#
# # Initialize SCQ model
# model = SCQ_model(input_dim=feature_dim, general_embed_dim=128, codebook_dim=32, codebook_num=5, descrete_loss=False, heads=8)
# # Compile model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
# # Train model and capture history
# history = model.fit(train_dataset, epochs=20, batch_size=batch_size)
# # Convert history to DataFrame and save as CSV
# history_df = pd.DataFrame(history.history)
# history_df.to_csv("test_tmp/model_history.csv", index=False)
#
# # Test model on new data correctly by iterating through batches
# decoded_outputs = []
# zq_outputs = []
# pj_outputs = []
#
# for batch in test_dataset:
#     output = model(batch)
#     decoded_outputs.append(output[0].numpy())
#     zq_outputs.append(output[1].numpy())
#     pj_outputs.append(output[2].numpy())
#
# # Save input and output arrays as .npy files
# os.makedirs('test_tmp', exist_ok=True)
# np.save("test_tmp/input_data.npy", X_test)  # Save the actual test data
# np.savez("test_tmp/output_data.npz",
#          decoded=np.vstack(decoded_outputs),
#          zq=np.vstack(zq_outputs),
#          pj=np.vstack(pj_outputs))
#
# print("Training complete. Model history saved as 'model_history.csv'.")
# print("Input and output arrays saved as 'input_data.npy' and 'output_data.npz'.")
# print("Shape of model outputs:", np.vstack(pj_outputs).shape)


