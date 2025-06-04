# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class SCQ_layer(layers.Layer):
    def __init__(self, num_embed, dim_embed, lambda_reg=1.0, proj_iter=10,
                 discrete_loss=False, beta_loss=0.25, reset_dead_codes=False,
                 usage_threshold=1e-3, reset_interval=5, **kwargs):
        """
        Soft Convex Quantization (SCQ) layer.

        Args:
            num_embed: Number of codebook vectors.
            dim_embed: Embedding dimension (for both encoder and codebook).
            lambda_reg: Regularization parameter in the SCQ objective.
            proj_iter: Number of iterations for the iterative simplex projection.
            discrete_loss: If True, commitment and codebook loss are calculated similar
                to VQ-VAE loss, with stop-gradient operation. If False, only the quantization error
                is calculated as the loss |Z_q - Z_e|**2.
            beta_loss: A float used in VQ loss. Only applicable if discrete_loss is True.
                Multiplied to (1-beta_loss)*commitment_loss + beta_loss*codebook_loss.
            reset_dead_codes: Whether to reset unused codebook vectors periodically.
            usage_threshold: Minimum usage threshold for codebook vectors to avoid being reset.
            reset_interval: Number of calls before checking and resetting dead codebooks.
        """
        super().__init__(**kwargs)
        self.num_embed = num_embed
        self.dim_embed = dim_embed
        self.lambda_reg = lambda_reg
        self.proj_iter = proj_iter
        self.discrete_loss = discrete_loss
        self.beta_loss = beta_loss
        self.epsilon = 1e-5

        # Codebook reset parameters
        self.call_count = 0
        self.reset_dead_codes = reset_dead_codes
        self.usage_threshold = usage_threshold
        self.reset_interval = reset_interval

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
            shape=(self.dim_embed, self.dim_embed),
            initializer='random_normal',
            trainable=True,
            name='scq_w')
        self.d_b = self.add_weight(
            shape=(self.dim_embed,),
            initializer='zeros',
            trainable=True,
            name='scq_b')
        # Usage tracking for codebook vectors
        self.code_usage = self.add_weight(
            shape=(self.num_embed,),
            initializer='zeros',
            trainable=False,
            name='code_usage',
            dtype=tf.float32
        )

    def call(self, inputs):
        """
        Forward pass of the SCQ layer.
        Args:
            inputs: Input tensor of shape (B, N, dim_embed), where B is the batch size,
                    N is the number of input features, and dim_embed is the embedding dimension.
        Returns:
            Zq: Quantized output tensor of shape (B, N, dim_embed).
            out_P_proj:
            loss: The total loss calculated during the forward pass.
            perplexity: The perplexity of the quantized output.

        """
        # 1. Project inputs to the embedding space and apply layer normalization.
        x = tf.matmul(inputs, self.d_w) + self.d_b  # (B, N, dim_embed)
        x = self.layernorm(x)

        input_shape = tf.shape(x)  # (B, N, dim_embed)
        flat_inputs = tf.reshape(x, [-1, self.dim_embed])  # (B*N, dim_embed)

        # add a small epsilon to avoid numerical issues # TODO added
        # flat_inputs = flat_inputs + tf.random.normal(tf.shape(flat_inputs), stddev=0.001)

        # 2. Compute hard VQ assignments as initialization.
        flat_detached = tf.stop_gradient(flat_inputs)
        x_norm_sq = tf.reduce_sum(flat_detached ** 2, axis=1, keepdims=True)  # (B*N, 1)
        codebook_norm_sq = tf.reduce_sum(self.embed_w ** 2, axis=0, keepdims=True)  # (1, num_embed)
        dot_product = tf.matmul(flat_detached, self.embed_w)  # (B*N, num_embed)
        distances = x_norm_sq + codebook_norm_sq - 2 * dot_product  # (B*N, num_embed)

        assign_indices = tf.argmin(distances, axis=1)  # (B*N,)
        P_tilde = tf.one_hot(assign_indices, depth=self.num_embed, dtype=tf.float32)  # (B*N, num_embed)
        P_tilde = tf.transpose(P_tilde)  # (num_embed, B*N)

        # Track codebook usage # TODO added
        if self.reset_dead_codes:
            # Update usage counts (moving average)
            batch_usage = tf.reduce_mean(tf.transpose(P_tilde), axis=0)  # (num_embed,)
            decay = 0.99  # Exponential moving average decay factor
            self.code_usage.assign(decay * self.code_usage + (1 - decay) * batch_usage)

            # Periodically check for dead codes and reset them
            self.call_count += 1
            if self.call_count % self.reset_interval == 0:
                self._reset_dead_codes(flat_inputs)

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
        out_P_proj = tf.transpose(P_proj)  # (B*N, num_embed)
        # save in a file for debugging
        # print out for debugging
        # tf.print("P_proj:", P_proj)
        # tf.print("P_proj min:", tf.reduce_min(P_proj))
        # tf.print("P_proj max:", tf.reduce_max(P_proj))
        # tf.print("P_proj shape:", tf.shape(P_proj))
        # tf.print("out_P_proj1:", out_P_proj)
        # tf.print("out_P_proj1 min:", tf.reduce_min(out_P_proj))
        # tf.print("out_P_proj1 max:", tf.reduce_max(out_P_proj))
        # tf.print("out_P_proj1 shape:", tf.shape(out_P_proj))

        out_P_proj = tf.reshape(out_P_proj, (input_shape[0], input_shape[1], -1))  # (B,N,num_embed)
        # tf.print("out_P_proj2:", out_P_proj)
        # tf.print("out_P_proj2 min:", tf.reduce_min(out_P_proj))
        # tf.print("out_P_proj2 max:", tf.reduce_max(out_P_proj))
        # tf.print("out_P_proj2 shape:", tf.shape(out_P_proj))

        # compute perplexity
        perplexity = self.compute_perplexity(out_P_proj)  # (B,N,num_embed)
        # tf.print("perplexity out_proj2:", perplexity)

        # average over the last dimension
        # out_P_proj = tf.reduce_mean(out_P_proj, axis=-1, keepdims=True)  # (B,N,1)
        # tf.print("out_P_proj3:", out_P_proj)
        # tf.print("out_P_proj3 min:", tf.reduce_min(out_P_proj))
        # tf.print("out_P_proj3 max:", tf.reduce_max(out_P_proj))
        # tf.print("out_P_proj3 shape:", tf.shape(out_P_proj))

        # # check if the shape of out_P_proj is (B,N,1)
        # if out_P_proj.shape[-1] != 1:
        #     raise ValueError(f"Expected out_P_proj shape to be (B,N,1), but got {out_P_proj.shape}")

        # 5. Reconstruct quantized output: Z_q = C * P_proj.
        Zq_flat = tf.matmul(C, P_proj)  # (dim_embed, B*N)
        Zq = tf.transpose(Zq_flat)  # (B*N, dim_embed)

        # 6. Calculate quantization loss and add regularization penalties
        # Calculate common regularization terms first
        p_mean = tf.reduce_mean(out_P_proj, axis=[0, 1])  # Average usage across batch (num_embed,)
        # Ensure probabilities sum to 1 for stable entropy calculation
        p_mean_normalized = p_mean / (tf.reduce_sum(p_mean) + self.epsilon)
        # Maximize entropy: encourages uniform cluster usage. Negative sign makes it a penalty.
        # Adding epsilon inside log avoids log(0).
        entropy_reg = -tf.reduce_sum(p_mean_normalized * tf.math.log(p_mean_normalized + self.epsilon))

        # Calculate cosine similarities between codebook vectors
        norm_codebook = tf.nn.l2_normalize(self.embed_w, axis=0)  # (dim_embed, num_embed)
        similarities = tf.matmul(tf.transpose(norm_codebook), norm_codebook)  # (num_embed, num_embed)
        # Create a mask to ignore self-similarities (diagonal)
        mask = tf.ones_like(similarities) - tf.eye(self.num_embed)
        masked_similarities = similarities * mask
        # Penalize positive cosine similarities between different codebook vectors
        # Encourages codebook vectors to be dissimilar (orthogonal ideally)
        diversity_loss = tf.reduce_mean(tf.nn.relu(masked_similarities))

        # Define weights for regularization terms (these can be tuned)
        alpha_entropy = 0.5  # Weight for entropy regularization penalty
        alpha_diversity = 0.5  # Weight for diversity penalty

        # Calculate the primary loss based on discrete_loss flag
        if not self.discrete_loss:
            # Original SCQ loss (quantization error)
            primary_loss = tf.reduce_mean((Zq - flat_inputs) ** 2)
        else:
            # VQ-VAE style loss
            commitment_loss = tf.reduce_mean((tf.stop_gradient(Zq) - flat_inputs) ** 2)
            codebook_loss = tf.reduce_mean((Zq - tf.stop_gradient(flat_inputs)) ** 2)  # Corrected VQ codebook loss
            primary_loss = (
                    (tf.cast(1 - self.beta_loss, tf.float32) * commitment_loss) +
                    (tf.cast(self.beta_loss, tf.float32) * codebook_loss)
            )

        # Combine primary loss with regularization penalties
        loss = primary_loss + alpha_entropy * entropy_reg + alpha_diversity * diversity_loss

        Zq = tf.reshape(Zq, input_shape)  # (B, N, dim_embed)
        self.add_loss(loss)  # Register the loss with the layer

        # Get hard indices and create one-hot representation # TODO added
        # hard_indices = tf.argmax(out_P_proj, axis=-1)  # (B,N)
        # cast to int32 for one-hot encoding
        # out_P_proj_cast = tf.cast(out_P_proj*10, tf.int32)  # (B,N,num_embed)
        # one_hot_indices = tf.one_hot(hard_indices, depth=self.num_embed)  # (B,N,num_embed)

        # 7. Calculate perplexity (similar to VQ layer)
        # flat_P_proj = tf.reshape(out_P_proj, [-1, self.num_embed])  # (B*N, num_embed)
        # avg_probs = tf.reduce_mean(flat_P_proj, axis=0)  # (num_embed,)
        # perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return Zq, out_P_proj, loss, perplexity

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

    def compute_perplexity(slef, out_P_proj):
        p_j = tf.reduce_mean(out_P_proj, axis=[0, 1])  # (B, N, K) -> (K,)
        p_j = tf.clip_by_value(p_j, 1e-10, 1 - (1e-9))
        p_j = p_j / tf.reduce_sum(p_j)  # Normalize to ensure sum to 1
        entropy = -tf.reduce_sum(p_j * tf.math.log(p_j) / tf.math.log(2.0))  # Entropy: -sum(p_j * log2(p_j))
        perplexity = tf.pow(2.0, entropy)  # Perplexity: 2^entropy
        return perplexity

    def layernorm(self, inputs):
        """
        Custom layer normalization over the last dimension.
        """
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.math.reduce_variance(inputs, axis=-1, keepdims=True)
        normed = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normed + self.beta

    def _reset_dead_codes(self, encoder_outputs):
        dead_codes = tf.where(self.code_usage < self.usage_threshold)
        num_dead = tf.shape(dead_codes)[0]
        if num_dead > 0:
            tf.print(f"Resetting {num_dead} dead codebook vectors")
            most_used_idx = tf.argsort(self.code_usage, direction='DESCENDING')[:tf.maximum(3, num_dead)]
            most_used = tf.gather(tf.transpose(self.embed_w), most_used_idx)
            batch_size = tf.shape(encoder_outputs)[0]
            for i in range(num_dead):
                dead_idx = tf.cast(dead_codes[i][0], tf.int32)
                if i % 2 == 0 and batch_size > 0:
                    random_idx = tf.random.uniform(shape=[], minval=0, maxval=batch_size, dtype=tf.int32)
                    new_vector = encoder_outputs[random_idx]
                else:
                    source_idx = i % tf.shape(most_used)[0]
                    source_vector = most_used[source_idx]
                    noise = tf.random.normal(shape=tf.shape(source_vector), stddev=0.5)
                    new_vector = source_vector + noise
                    new_vector = new_vector / (tf.norm(new_vector) + 1e-8) * tf.norm(source_vector)
                self.embed_w[:, dead_idx].assign(tf.reshape(new_vector, [self.dim_embed]))
                self.code_usage[dead_idx].assign(0.2)


class SCQ1DAutoEncoder(keras.Model):
    def __init__(self, input_dim, num_embeddings, embedding_dim, commitment_beta,
                 scq_params, initial_codebook=None, cluster_lambda=1.0):
        super().__init__()
        self.input_dim = input_dim  # e.g., (1024,)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_beta = commitment_beta

        # weight on the new cluster‐consistency loss
        self.cluster_lambda = cluster_lambda

        # Encoder: Dense layers to compress 1024 features to embedding_dim
        self.encoder = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=self.input_dim),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.embedding_dim, activation='linear')
        ])

        # SCQ Layer for quantization
        self.scq_layer = SCQ_layer(num_embed=self.num_embeddings, dim_embed=self.embedding_dim,
                                   beta_loss=self.commitment_beta, **scq_params)

        # Initialize codebook if provided
        if initial_codebook is not None:
            # Ensure the shape matches
            expected_shape = (self.embedding_dim, self.num_embeddings)
            if initial_codebook.shape == expected_shape:
                self.scq_layer.embed_w.assign(initial_codebook)
            else:
                tf.print(
                    f"Warning: Initial codebook shape {initial_codebook.shape} doesn't match expected shape {expected_shape}. Using default initialization.")

        # Decoder: Dense layers to reconstruct from embedding_dim to 1024 features
        self.decoder = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.embedding_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.input_dim[0], activation='linear')  # output shape (B, 1024)
        ])

        # Loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.perplexity_tracker = keras.metrics.Mean(name="perplexity")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker,
            self.perplexity_tracker,
        ]


    def call(self, inputs, training=False):
        if isinstance(inputs, tuple):
            x, labels = inputs
        else:
            x, labels = inputs, None
        # Encoder: Compress input to embedding space
        x = self.encoder(x)  # (batch_size, embedding_dim)
        x = tf.expand_dims(x, axis=1)  # (batch_size, 1, embedding_dim)
        # SCQ: Quantize the embedding
        Zq, out_P_proj, vq_loss, perplexity = self.scq_layer(x)
        # Decoder: Reconstruct from quantized embedding
        y = tf.squeeze(Zq, axis=1)  # (batch_size, embedding_dim)
        output = self.decoder(y)  # (batch_size, 1024)
        return output, Zq, out_P_proj, vq_loss, perplexity

    # # Method to get just the latent sequence and one-hot encodings for inference
    def encode_(self, inputs):
        if isinstance(inputs, tuple):
            x, labels = inputs
        else:
            x, labels = inputs, None
        # Encoder: Compress input to embedding space
        x = self.encoder(x)  # (batch_size, embedding_dim)
        x = tf.expand_dims(x, axis=1)  # (batch_size, 1, embedding_dim)
        # SCQ: Quantize the embedding
        Zq, out_P_proj, vq_loss, perplexity = self.scq_layer(x)
        return Zq, out_P_proj, vq_loss, perplexity

    def train_step(self, data):
        # unpack features and (optional) labels
        if isinstance(data, tuple):
            x, labels = data
        else:
            x, labels = data, None

        with tf.GradientTape() as tape:
            reconstruction, Zq, _, vq_loss, perplexity = self(x, training=True)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(tf.math.squared_difference(x, reconstruction))

            # Feature matching loss
            with tape.stop_recording():
                _, mid_features, _, _, _ = self(reconstruction, training=False)
                orig_mid_features = Zq
            feature_loss = tf.reduce_mean(tf.math.squared_difference(orig_mid_features, mid_features))

            # Total loss
            total_loss = recon_loss + vq_loss + 0.1 * feature_loss

            # Usage penalty
            usage_penalty = tf.cond(
                perplexity < self.num_embeddings * 0.5,
                lambda: 0.5 * (self.num_embeddings * 0.5 - perplexity),
                lambda: 0.0
            )
            total_loss += usage_penalty

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        self.perplexity_tracker.update_state(perplexity)

        # Log perplexity every 100 steps
        step = self.optimizer.iterations
        tf.cond(
            tf.equal(tf.math.floormod(step, 100), 0),
            lambda: tf.print("Step:", step, "Perplexity:", perplexity, "Target:", self.num_embeddings * 0.5),
            lambda: tf.no_op()
        )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
            y = data[1] if len(data) > 1 else data[0]
        else:
            x = data

        reconstruction, quantized, _, vq_loss, perplexity = self(x, training=False)
        recon_loss = tf.reduce_mean(tf.math.squared_difference(x, reconstruction))
        total_loss = recon_loss + vq_loss + self.commitment_beta * tf.reduce_mean(
            tf.math.squared_difference(x, reconstruction))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        self.perplexity_tracker.update_state(perplexity)

        return {m.name: m.result() for m in self.metrics}


########### Mixture of Experts (MoE) model ###########
class SparseDispatcher:
    """Helper for dispatching inputs to experts and combining expert outputs."""
    def __init__(self, num_experts, gates):
        # gates: [batch, num_experts] float tensor
        self.num_experts = num_experts
        self.gates = gates  # shape [B, E]
        # Find nonzero gate entries
        indices = tf.where(gates > 0)
        # Sort by expert_idx then batch_idx for consistency
        sort_order = tf.argsort(indices[:,1] * tf.cast(tf.shape(gates)[0], indices.dtype) + indices[:,0])
        sorted_indices = tf.gather(indices, sort_order)
        self.batch_index = sorted_indices[:,0]
        self.expert_index = sorted_indices[:,1]
        # Count number of samples per expert
        self.part_sizes = tf.reduce_sum(tf.cast(gates > 0, tf.int32), axis=0)
        # Extract the nonzero gate values
        self.nonzero_gates = tf.gather_nd(gates, sorted_indices)

    def dispatch(self, inputs):
        """inputs: [batch, ...] -> Returns list of [num_samples_i, ...]"""
        inputs_expanded = tf.gather(inputs, self.batch_index)
        parts = tf.split(inputs_expanded, self.part_sizes, axis=0)
        return parts

    def combine(self, expert_outputs, multiply_by_gates=True):
        """expert_outputs: list of [num_samples_i, output_dim] -> Returns [batch, output_dim]"""
        stitched = tf.concat(expert_outputs, axis=0)
        if multiply_by_gates:
            stitched = stitched * tf.expand_dims(self.nonzero_gates, axis=1)
        batch_size = tf.shape(self.gates)[0]
        combined = tf.math.unsorted_segment_sum(stitched, self.batch_index, batch_size)
        return combined


class Expert(layers.Layer):
    """A binary prediction expert with added complexity using GLU."""

    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
        self.fc2 = layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.nn.sigmoid(x)

# class Expert(layers.Layer):
#     """Enhanced binary prediction expert with dropout and more layers."""
#     def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.2):
#         super().__init__()
#         self.fc1 = layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
#         self.dropout1 = layers.Dropout(dropout_rate)
#         self.fc2 = layers.Dense(hidden_dim // 2, activation='relu')
#         self.dropout2 = layers.Dropout(dropout_rate)
#         self.fc3 = layers.Dense(output_dim)
#
#     def call(self, x, training=False):
#         x = self.fc1(x)
#         x = self.dropout1(x, training=training)
#         x = self.fc2(x)
#         x = self.dropout2(x, training=training)
#         x = self.fc3(x)
#         return tf.nn.sigmoid(x)


class MixtureOfExperts(layers.Layer):
    """Mixture of Experts layer with optional learned gating network."""

    def __init__(self, input_dim, hidden_dim, num_experts, use_provided_gates=True,
                 gating_hidden_dim=64, top_k=None):
        super().__init__()
        self.num_experts = num_experts
        self.use_provided_gates = use_provided_gates
        self.top_k = top_k  # If set, only route to top_k experts
        self.experts = [Expert(input_dim, hidden_dim) for _ in range(num_experts)]

        # Add a learned gating network if we might use it
        if not use_provided_gates:
            self.gating_network = tf.keras.Sequential([
                layers.Dense(gating_hidden_dim, activation='relu', input_shape=(input_dim,)),
                layers.Dense(num_experts, activation='softmax')
            ])

    def call(self, inputs, training=False):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            x, gates = inputs
        else:
            x = inputs
            gates = None

        # Determine the gates to use
        if gates is None or not self.use_provided_gates:
            if hasattr(self, 'gating_network'):
                # Use learned gating if available
                gates = self.gating_network(x)
            else:
                # Fall back to uniform distribution
                batch_size = tf.shape(x)[0]
                gates = tf.ones([batch_size, self.num_experts]) / self.num_experts

        # Apply top-k gating if configured
        if self.top_k is not None and self.top_k < self.num_experts:
            # Get values and indices of top-k gate values
            _, top_k_indices = tf.math.top_k(gates, k=self.top_k)
            # Create a mask for the top_k gates (1 for top-k, 0 for others)
            mask = tf.reduce_sum(
                tf.one_hot(top_k_indices, depth=self.num_experts),
                axis=1
            )
            # Zero out non-top-k gates and renormalize
            gates = gates * mask
            gates = gates / (tf.reduce_sum(gates, axis=-1, keepdims=True) + 1e-10)

        # Create dispatcher with these gates
        dispatcher = SparseDispatcher(self.num_experts, gates)

        # Dispatch inputs to experts
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = [
            expert(inp)
            for expert, inp in zip(self.experts, expert_inputs)
        ]

        # Combine expert outputs
        combined_outputs = dispatcher.combine(expert_outputs)

        return {
            'prediction': combined_outputs,
            'gates': gates,
            'expert_outputs': expert_outputs if training else None
        }

class MoEModel(tf.keras.Model):
    """Model wrapping the MoE layer with optional learned gating."""
    def __init__(self, input_dim, hidden_dim, num_experts, use_provided_gates=True,
                 gating_hidden_dim=64, top_k=None):
        super().__init__()
        self.use_provided_gates = use_provided_gates
        self.moe_layer = MixtureOfExperts(
            input_dim,
            hidden_dim,
            num_experts,
            use_provided_gates=use_provided_gates,
            gating_hidden_dim=gating_hidden_dim,
            top_k=top_k
        )

    def call(self, inputs, training=False):
        # Handle both cases: with and without provided gates
        moe_outputs = self.moe_layer(inputs, training=training)
        return moe_outputs if not training else moe_outputs['prediction']

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            raise ValueError("Training data must include both inputs and labels")

        with tf.GradientTape() as tape:
            # Forward pass - handle both input types
            if isinstance(x, tuple) and len(x) == 2:
                # Input includes features and gates
                inputs, gates = x
                outputs = self.moe_layer((inputs, gates), training=True)
            else:
                # Input is just features, gates will be computed by gating network
                outputs = self.moe_layer(x, training=True)

            predictions = outputs['prediction']

            # Compute loss
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

            # Add expert load balancing loss
            gates = outputs['gates']
            expert_usage = tf.reduce_mean(gates, axis=0)
            load_balancing_loss = tf.reduce_sum(expert_usage * tf.math.log(expert_usage + 1e-10)) * 0.01
            total_loss = loss + load_balancing_loss

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss, 'load_balancing_loss': load_balancing_loss})
        return results

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            raise ValueError("Test data must include both inputs and labels")

        # Forward pass - handle both input types
        if isinstance(x, tuple) and len(x) == 2:
            # Input includes features and gates
            inputs, gates = x
            outputs = self.moe_layer((inputs, gates), training=False)
        else:
            # Input is just features, gates will be computed by gating network
            outputs = self.moe_layer(x, training=False)

        predictions = outputs['prediction']

        # Compute loss
        loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def predict(self, inputs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            # Input includes features and gates
            inputs, gates = inputs
            outputs = self.moe_layer((inputs, gates), training=False)
        else:
            # Input is just features, gates will be computed by gating network
            outputs = self.moe_layer(inputs, training=False)

        predictions = outputs['prediction']
        return predictions


def visualize_dataset_analysis(features, labels, cluster_probs, method='pca', raw_dot_plot=False, feature_indices=None):
        """
        Visualize the relationship between features, labels, and cluster probabilities.

        Args:
            features: TensorFlow tensor containing feature vectors
            labels: TensorFlow tensor containing labels
            cluster_probs: TensorFlow tensor containing cluster probabilities
            method: Dimensionality reduction method ('umap', 'tsne', or 'pca')
            raw_dot_plot: If True, plot raw feature values directly
            feature_indices: Indices of features to plot (tuple of 2 indices)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Convert tensors to numpy arrays
        features_np = features.numpy()
        labels_np = labels.numpy().flatten()
        cluster_probs_np = cluster_probs.numpy()

        # Compute dominant cluster for each sample
        dominant_clusters = np.argmax(cluster_probs_np, axis=1)

        if raw_dot_plot:
            # Plot raw feature values directly without dimensionality reduction
            feature_indices = feature_indices or (0, 1)  # Default to first two features

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Scatter by true label
            scatter1 = ax1.scatter(
                features_np[:, feature_indices[0]],
                features_np[:, feature_indices[1]],
                c=labels_np,
                cmap='viridis',
                s=10,
                alpha=0.6
            )
            ax1.set_title(f'Raw Features: Colored by Label')
            ax1.set_xlabel(f'Feature {feature_indices[0]}')
            ax1.set_ylabel(f'Feature {feature_indices[1]}')
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Label')

            # Scatter by dominant cluster
            n_clusters = cluster_probs_np.shape[1]
            cmap_clusters = plt.cm.get_cmap('tab20b', n_clusters) if n_clusters <= 20 else plt.cm.get_cmap('gist_ncar', n_clusters)

            scatter2 = ax2.scatter(
                features_np[:, feature_indices[0]],
                features_np[:, feature_indices[1]],
                c=dominant_clusters,
                cmap=cmap_clusters,
                s=10,
                alpha=0.6
            )
            ax2.set_title(f'Raw Features: Colored by Cluster')
            ax2.set_xlabel(f'Feature {feature_indices[0]}')
            ax2.set_ylabel(f'Feature {feature_indices[1]}')
            cbar2 = plt.colorbar(scatter2, ax=ax2, ticks=range(n_clusters))
            cbar2.set_label('Cluster ID')

            plt.tight_layout()
            plt.savefig(f'raw_feature_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            return

        # Original dimensionality reduction visualization
        try:
            if method == 'umap':
                import umap
                reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
                features_2d = reducer.fit_transform(features_np)
                method_name = "UMAP"
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                features_2d = reducer.fit_transform(features_np)
                method_name = "t-SNE"
            else:  # default to PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
                features_2d = reducer.fit_transform(features_np)
                method_name = "PCA"
        except ImportError:
            # Fall back to PCA if the requested method is not available
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            features_2d = reducer.fit_transform(features_np)
            method_name = "PCA"
            print(f"Warning: {method} not available, falling back to PCA")

        # Prepare subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter by true label
        scatter1 = ax1.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels_np,
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        ax1.set_title(f'Samples by Label ({method_name})')
        ax1.set_xlabel(f'{method_name} Component 1')
        ax1.set_ylabel(f'{method_name} Component 2')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Label')

        # Scatter by dominant cluster with improved discrete colormap
        n_clusters = cluster_probs_np.shape[1]
        # Use a better colormap for distinguishing clusters
        cmap_clusters = plt.cm.get_cmap('tab20b', n_clusters) if n_clusters <= 20 else plt.cm.get_cmap('gist_ncar', n_clusters)

        # Add confidence information - marker size based on max probability
        max_probs = np.max(cluster_probs_np, axis=1)
        marker_sizes = 10 + 40 * max_probs  # Scale confidence to marker size

        scatter2 = ax2.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=dominant_clusters,
            cmap=cmap_clusters,
            s=marker_sizes,
            alpha=0.6,
            edgecolors='none'
        )
        ax2.set_title(f'Samples by Dominant Cluster ({method_name})')
        ax2.set_xlabel(f'{method_name} Component 1')
        ax2.set_ylabel(f'{method_name} Component 2')
        cbar2 = plt.colorbar(scatter2, ax=ax2, ticks=range(n_clusters))
        cbar2.set_label('Cluster ID')

        plt.tight_layout()
        plt.savefig(f'cluster_analysis_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot overall label and cluster probability distributions
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Label distribution
        unique, counts = np.unique(labels_np, return_counts=True)
        ax[0].bar(unique.astype(int), counts, color='skyblue')
        ax[0].set_xlabel('Label')
        ax[0].set_ylabel('Count')
        ax[0].set_title('Label Distribution')

        # Cluster probability distribution
        cluster_prob_sums = np.sum(cluster_probs_np, axis=0)
        ax[1].bar(np.arange(len(cluster_prob_sums)), cluster_prob_sums, color='salmon')
        ax[1].set_xlabel('Cluster')
        ax[1].set_ylabel('Sum of Probabilities')
        ax[1].set_title('Cluster Probability Distribution')

        plt.tight_layout()
        plt.savefig('dataset_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        # TODO show the whole dataset as a dot plot
        # Plot the dataset as a dot plot of the first two features
        plt.figure(figsize=(10, 8))
        plt.scatter(features_np[:, 0], features_np[:, 1], c=labels_np, cmap='coolwarm',
                    s=10, alpha=0.7, edgecolors='none')
        plt.colorbar(label='Label')
        plt.title('Dot Plot of First Two Features')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.grid(linestyle='--', alpha=0.6)
        plt.savefig('feature_dot_plot.png', dpi=300, bbox_inches='tight')
        plt.show()


# # Example usage
# if __name__ == "__main__":
#     # Generate dummy dataset with clustered data and labels for training
#     num_train_samples = 8000
#     num_test_samples = 2000
#     feature_dim = 64
#     num_clusters = 32
#
#     # Function to generate dataset with specified parameters
#     def generate_dataset(num_samples, feature_dim, num_clusters, epsilon=0.1):
#         # Compute samples per cluster
#         base_count = num_samples // num_clusters
#         counts = [base_count + (1 if i < num_samples % num_clusters else 0) for i in range(num_clusters)]
#
#         features_list = []
#         labels_list = []
#         cluster_probs_list = []
#         distributions = ['normal', 'uniform', 'gamma', 'poisson']
#
#         for c in range(num_clusters):
#             cluster_count = counts[c]
#             n0 = cluster_count // 2
#             n1 = cluster_count - n0
#             dist = distributions[(2 * c) % len(distributions)]
#             print(f"Cluster {c}: {dist} distribution, {n0} samples 0, {n1} samples 1")
#
#             if dist == 'normal':
#                 features_0 = tf.random.normal([n0, feature_dim], mean=c, stddev=1.0)
#             elif dist == 'uniform':
#                 features_0 = tf.random.uniform([n0, feature_dim], minval=0, maxval=1)
#             elif dist == 'gamma':
#                 features_0 = tf.random.gamma([n0, feature_dim], alpha=2.0, beta=1.0)
#             elif dist == 'poisson':
#                 features_0 = tf.cast(tf.random.poisson([n0, feature_dim], lam=3), tf.float32)
#             else:
#                 features_0 = tf.random.normal([n0, feature_dim], mean=c, stddev=1.0)
#
#             if dist == 'normal':
#                 features_1 = tf.random.normal([n1, feature_dim], mean=c+0.5, stddev=1.5)
#             elif dist == 'uniform':
#                 features_1 = tf.random.uniform([n1, feature_dim], minval=1, maxval=2)
#             elif dist == 'gamma':
#                 features_1 = tf.random.gamma([n1, feature_dim], alpha=5.0, beta=2.0)
#             elif dist == 'poisson':
#                 features_1 = tf.cast(tf.random.poisson([n1, feature_dim], lam=6), tf.float32)
#             else:
#                 features_1 = tf.random.normal([n1, feature_dim], mean=c+0.5, stddev=1.5)
#
#             features_i = tf.concat([features_0, features_1], axis=0)
#             labels_i = tf.concat([tf.zeros([n0, 1], tf.int32), tf.ones([n1, 1], tf.int32)], axis=0)
#             features_list.append(features_i)
#             labels_list.append(labels_i)
#
#             # Generate random cluster probabilities per sample
#             cluster_indices = tf.fill([cluster_count], c)
#             lam_value = tf.maximum(tf.cast(c, tf.float32) + 1.0, 1.0)
#             noise = tf.cast(tf.random.poisson([cluster_count, num_clusters], lam=lam_value), tf.float32)
#             noise = noise + epsilon
#             probs = noise / (tf.reduce_sum(noise, axis=1, keepdims=True))
#             alpha = tf.random.uniform([cluster_count, 1], minval=0.5, maxval=0.8)
#             probs = (1 - alpha) * probs + alpha * tf.one_hot(cluster_indices, num_clusters)
#             probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
#             cluster_probs_list.append(probs)
#
#         features = tf.concat(features_list, axis=0)
#         labels = tf.concat(labels_list, axis=0)
#         cluster_probs = tf.concat(cluster_probs_list, axis=0)
#
#         # Shuffle dataset
#         indices = tf.random.shuffle(tf.range(tf.shape(features)[0]))
#         features = tf.gather(features, indices)
#         labels = tf.gather(labels, indices)
#         cluster_probs = tf.gather(cluster_probs, indices)
#
#         return features, labels, cluster_probs
#
#     # Generate training dataset
#     print("\nGenerating training dataset...")
#     train_features, train_labels, train_cluster_probs = generate_dataset(
#         num_train_samples, feature_dim, num_clusters)
#
#     # Generate separate test dataset
#     print("\nGenerating test dataset...")
#     test_features, test_labels, test_cluster_probs = generate_dataset(
#         num_test_samples, feature_dim, num_clusters, epsilon=1)
#
#     print(f"\nTraining labels min: {tf.reduce_min(train_labels)}, max: {tf.reduce_max(train_labels)}")
#     print(f"Training labels head: {train_labels[:5]}")
#     print(f"Test labels min: {tf.reduce_min(test_labels)}, max: {tf.reduce_max(test_labels)}")
#     print(f"Test labels head: {test_labels[:5]}")
#
#     # Visualize training dataset
#     print("\nVisualizing training dataset...")
#     visualize_dataset_analysis(train_features, train_labels, train_cluster_probs,
#                               raw_dot_plot=False, method='pca', feature_indices=(0, 1))
#
#     # Create training dataset
#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         ((train_features, train_cluster_probs), train_labels)
#     ).shuffle(1000).batch(64)
#
#     # Create test dataset (no need to shuffle extensively)
#     test_dataset = tf.data.Dataset.from_tensor_slices(
#         ((test_features, test_cluster_probs), test_labels)
#     ).batch(64)
#
#     # Print info about test dataset
#     for features, labels in test_dataset.take(1):
#         print(f"\nTest features shape: {features[0].shape}, Test labels shape: {labels.shape}")
#         print(f"Test features head: {features[0][:3]}")
#         print(f"Test labels head: {labels[:3]}")
#
#     # Create and compile model
#     model = MoEModel(feature_dim, hidden_dim=8, num_experts=num_clusters, use_provided_gates=True)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=['accuracy'],
#     )
#
#     # Train model
#     print("\nTraining model...")
#     model.fit(train_dataset, epochs=20)
#
#     # Evaluate on independent test set
#     print("\nEvaluating on independent test set...")
#     eval_results = model.evaluate(test_dataset)
#     print(f"Evaluation results: {eval_results}")
#
#     # Predict and analyze
#     # test_batch = next(iter(test_dataset.take(1)))
#     # predictions = model(test_batch[0], training=False)
#     # print(f"Sample predictions shape: {predictions['prediction'].shape}")
#     # print(f"Gate activations: {tf.reduce_mean(predictions['gates'], axis=0)}")
#
#     # Test on a single sample
#     for (feat, cluster_prob), label in test_dataset.unbatch().take(1):
#         feat = tf.expand_dims(feat, 0)
#         cluster_prob = tf.expand_dims(cluster_prob, 0)
#         output = model((feat, cluster_prob), training=False)
#         print("Single sample prediction:", output['prediction'].numpy()[0], "True label:", label.numpy())
#         print("Gate activations:", tf.reduce_mean(output['gates'], axis=0).numpy())
#
#     # Optional: Visualize test dataset
#     print("\nVisualizing test dataset...")
#     visualize_dataset_analysis(test_features, test_labels, test_cluster_probs,
#                               raw_dot_plot=False, method='pca', feature_indices=(0, 1))

class BinaryMLP(tf.keras.Model):
    def __init__(self, input_dim=1024):
        super(BinaryMLP, self).__init__()
        # First hidden layer
        self.dense1 = tf.keras.layers.Dense(
            units=512, activation='relu', name='dense_1'
        )
        self.dropout1 = tf.keras.layers.Dropout(
            rate=0.5, name='dropout_1'
        )
        # Second hidden layer
        self.dense2 = tf.keras.layers.Dense(
            units=256, activation='relu', name='dense_2'
        )
        self.dropout2 = tf.keras.layers.Dropout(
            rate=0.5, name='dropout_2'
        )
        # Output layer for binary classification
        self.output_layer = tf.keras.layers.Dense(
            units=1, activation='sigmoid', name='output_layer'
        )

    def call(self, inputs, training=False):
        # If inputs come in as (features, labels), unpack and ignore labels
        if isinstance(inputs, (tuple, list)):
            x, _ = inputs
        else:
            x = inputs

        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_layer(x)

# # Example usage
# if __name__ == "__main__":
#     # Generate dummy dataset
#     num_samples = 1000
#     input_dim = 1024
#     features = tf.random.normal((num_samples, input_dim))
#     labels = tf.random.uniform((num_samples,), minval=0, maxval=2, dtype=tf.int32)
#
#     # Create and compile model
#     model = BinaryMLP(input_dim=input_dim)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     # Train model
#     model.fit(features, labels, epochs=5, batch_size=32)
#     # Evaluate model
#     loss, accuracy = model.evaluate(features, labels)
#     print(f"Loss: {loss}, Accuracy: {accuracy}")
#     # Predict
#     predictions = model.predict(features)
#     print(f"Predictions shape: {predictions.shape}")
#     print(f"Predictions head: {predictions[:5]}")


class TransformerBlock(layers.Layer):
    """Single Transformer encoder block."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        # Self-attention block
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        # Feed-forward block
        ffn_output = self.ffn(out1, training=training)
        return self.norm2(out1 + ffn_output)


import tensorflow as tf
from tensorflow.keras import layers, Model

class TransformerBlock(layers.Layer):
    """Single Transformer encoder block."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        # Self-attention + residual + norm
        attn_output = self.attn(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        # Feed-forward + residual + norm
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x + ffn_output)


class TabularTransformer(Model):
    """
    Transformer-based classifier for tabular data.
    Applies a downsampling pool to reduce sequence length and avoid OOM.
    """
    def __init__(
        self,
        input_dim=1024,
        embed_dim=64,
        num_heads=8,
        ff_dim=256,
        num_layers=4,
        dropout_rate=0.1,
        pool_size=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        # Project each scalar feature to an embedding
        self.feature_embedding = layers.Dense(embed_dim, name='feature_embedding')
        # Downsample sequence length: 1024 -> 1024/pool_size
        self.pool = layers.MaxPool1D(pool_size=pool_size, name='sequence_pool')
        reduced_seq_len = input_dim // pool_size

        # Positional embeddings for reduced tokens
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, reduced_seq_len, embed_dim),
            initializer='random_normal'
        )
        # Transformer encoder stack
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        # Pool & classification head
        self.global_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')
        self.dropout = layers.Dropout(dropout_rate, name='dropout_final')
        self.classifier = layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs, training=False):
        # Unpack if inputs come as (features, labels)
        if isinstance(inputs, (tuple, list)):
            x, _ = inputs
        else:
            x = inputs

        # shape -> (batch, input_dim, 1)
        x = tf.expand_dims(x, axis=-1)
        # Embed features -> (batch, input_dim, embed_dim)
        x = self.feature_embedding(x)
        # Downsample tokens -> (batch, reduced_seq_len, embed_dim)
        x = self.pool(x)
        # Add positional embeddings
        x = x + self.pos_embedding

        # Transformer encoder stack
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Pool over tokens -> (batch, embed_dim)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)

# # Example usage
# if __name__ == "__main__":
#     num_samples = 1000
#     input_dim = 1024
#     features = tf.random.normal((num_samples, input_dim))
#     labels = tf.random.uniform((num_samples,), maxval=2, dtype=tf.int32)
#
#     model = TabularTransformer(input_dim=input_dim, pool_size=4)
#     model.build(input_shape=(None, input_dim))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
#
#     model.fit(features, labels, epochs=5, batch_size=32)
#     loss, accuracy = model.evaluate(features, labels)
#     print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#     # Predict
#     predictions = model.predict(features)
#     print(f"Predictions shape: {predictions.shape}")
#     print(f"Predictions head: {predictions[:5]}")


class EmbeddingCNN(Model):
    """
    1D-CNN + MLP head for binary classification on fixed-length embeddings.
    """

    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__(name='Embedding_CNN')
        self.input_dim = input_dim

        # Reshape flat embedding to sequence (length=input_dim, channels=1)
        self.reshape_layer = layers.Reshape((input_dim, 1), name='reshape')

        # Convolutional blocks
        self.conv1 = layers.Conv1D(64, 5, padding='same', activation='relu', name='conv1')
        self.pool1 = layers.MaxPool1D(2, name='pool1')

        self.conv2 = layers.Conv1D(128, 5, padding='same', activation='relu', name='conv2')
        self.pool2 = layers.MaxPool1D(2, name='pool2')

        # Flatten and MLP head
        self.flatten = layers.Flatten(name='flatten')
        self.dense = layers.Dense(64, activation='relu', name='dense')
        self.dropout = layers.Dropout(dropout_rate, name='dropout')

        # Final binary output
        self.output_layer = layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs, training=False):
        # Unpack if inputs come as (features, labels)
        if isinstance(inputs, (tuple, list)):
            x, _ = inputs
        else:
            x = inputs

        # Now safe to reshape just the feature tensor
        x = self.reshape_layer(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

# # # Example usage
# if __name__ == "__main__":
#     num_samples = 1000
#     input_dim = 1024
#     features = tf.random.normal((num_samples, input_dim))
#     labels = tf.random.uniform((num_samples,), maxval=2, dtype=tf.int32)
#     model = EmbeddingCNN(input_dim=input_dim, dropout_rate=0.5)
#     model.build(input_shape=(None, input_dim))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
#     model.fit(features, labels, epochs=5, batch_size=32)
#     loss, accuracy = model.evaluate(features, labels)
#     print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#     # Predict
#     predictions = model.predict(features)
#     print(f"Predictions shape: {predictions.shape}")
#     print(f"Predictions head: {predictions[:5]}")


# mixture_of_experts.py
"""
Mixture‑of‑Experts implementation supporting **soft** and **hard** clustering.

* **Soft clustering** (probabilistic gates) is used **during inference** to fuse the
  parameters of all experts into a *virtual* expert.
* **Hard clustering** (one‑hot gates) is used **during training**; each sample
  activates exactly one expert so gradients flow only through the selected expert.

Key ideas
---------
1.  **SparseDispatcher** routes inputs to experts and fuses outputs.  It now accepts
    either soft or hard gates transparently.
2.  **Expert** is a light two‑layer perceptron ending in a sigmoid.
3.  **MixtureOfExperts**
    * consumes `inputs` **and** a *soft* clustering vector (`gates_soft`).
    * converts to hard gates (`gates_hard`) with `tf.one_hot(tf.argmax(...))` when
      `hard_gating=True`.
    * during inference (`hard_gating=False`) the *soft* gates are used to create a
      **parameter‑blended virtual expert**: every weight matrix and bias vector is
      a convex combination of the corresponding parameters of the individual
      experts.
4.  **MoEModel** overrides `train_step` / `test_step` so that
    * training  → `hard_gating=True`
    * inference → `hard_gating=False`

The code is ready to run in a standard TensorFlow‑2 environment.
"""


class Expert(layers.Layer):
    """A binary prediction expert with added complexity."""

    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout_rate=0.2):
        super().__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
        self.dropout1 = layers.Dropout(dropout_rate)
        self.fc2 = layers.Dense(hidden_dim // 2, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.fc3 = layers.Dense(output_dim)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        x = self.fc3(x)
        return tf.nn.sigmoid(x)


class EnhancedMixtureOfExperts(layers.Layer):
    """
    Enhanced Mixture of Experts layer that uses cluster assignments.

    This implementation eliminates the need for a SparseDispatcher by:
    - During training: Using hard clustering to train specific experts
    - During inference: Using soft clustering to mix the experts' weights
    """

    def __init__(self, input_dim, hidden_dim, num_experts, output_dim=1,
                 use_hard_clustering=True, dropout_rate=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.use_hard_clustering = use_hard_clustering
        self.output_dim = output_dim

        # Create n experts
        self.experts = [
            Expert(input_dim, hidden_dim, output_dim, dropout_rate)
            for _ in range(num_experts)
        ]

    def convert_to_hard_clustering(self, soft_clusters):
        """Convert soft clustering values to hard clustering (one-hot encoding)"""
        # Get the index of the maximum value for each sample
        hard_indices = tf.argmax(soft_clusters, axis=1)
        # Convert to one-hot encoding
        return tf.one_hot(hard_indices, depth=self.num_experts)

    def call(self, inputs, training=False):
        # Unpack inputs
        if isinstance(inputs, tuple) and len(inputs) == 2:
            x, soft_cluster_probs = inputs
        else:
            raise ValueError("Inputs must include both features and clustering values")

        batch_size = tf.shape(x)[0]

        # Convert to hard clustering during training if requested
        if training and self.use_hard_clustering:
            clustering = self.convert_to_hard_clustering(soft_cluster_probs)
        else:
            clustering = soft_cluster_probs

        # Initialize output tensor
        combined_output = tf.zeros([batch_size, self.output_dim])

        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get the weight for this expert for each sample in the batch
            expert_weights = clustering[:, i:i + 1]  # Shape: [batch_size, 1]

            # Only compute outputs for samples with non-zero weights
            # to save computation during training with hard clustering
            if training and self.use_hard_clustering:
                # Find samples assigned to this expert
                assigned_indices = tf.where(expert_weights[:, 0] > 0)[:, 0]

                if tf.size(assigned_indices) > 0:
                    # Get assigned samples
                    assigned_x = tf.gather(x, assigned_indices)

                    # Get expert output for assigned samples
                    expert_output = expert(assigned_x, training=training)

                    # Use scatter_nd to place results back into full batch tensor
                    indices = tf.expand_dims(assigned_indices, axis=1)
                    updates = expert_output
                    combined_output += tf.scatter_nd(indices, updates, [batch_size, self.output_dim])
            else:
                # During inference or when using soft clustering:
                # Compute expert output for all samples
                expert_output = expert(x, training=training)

                # Weight the output by the clustering values
                weighted_output = expert_output * expert_weights

                # Add to combined output
                combined_output += weighted_output

        return combined_output


class EnhancedMoEModel(tf.keras.Model):
    """Complete model wrapping the Enhanced MoE layer"""

    def __init__(self, input_dim, hidden_dim, num_experts, output_dim=1,
                 use_hard_clustering=True, dropout_rate=0.2):
        super().__init__()
        self.use_hard_clustering = use_hard_clustering
        self.moe_layer = EnhancedMixtureOfExperts(
            input_dim,
            hidden_dim,
            num_experts,
            output_dim=output_dim,
            use_hard_clustering=use_hard_clustering,
            dropout_rate=dropout_rate
        )

    def call(self, inputs, training=False):
        return self.moe_layer(inputs, training=training)

    def train_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            # Unpack the data
            x, y = data

            # Ensure x contains both inputs and clustering values
            if not (isinstance(x, tuple) and len(x) == 2):
                raise ValueError("Input must be a tuple of (features, clustering)")
        else:
            raise ValueError("Training data must include both inputs and labels")

        # Unpack inputs
        inputs, soft_cluster_vector = x

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self(x, training=True)

            # Compute loss
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

            # Add expert load balancing loss if using soft clustering
            if not self.use_hard_clustering:
                expert_usage = tf.reduce_mean(soft_cluster_vector, axis=0)
                load_balancing_loss = tf.reduce_sum(expert_usage * tf.math.log(expert_usage + 1e-10)) * 0.01
                loss += load_balancing_loss

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def test_step(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            # Unpack the data
            x, y = data

            # Ensure x contains both inputs and clustering values
            if not (isinstance(x, tuple) and len(x) == 2):
                raise ValueError("Input must be a tuple of (features, clustering)")
        else:
            raise ValueError("Test data must include both inputs and labels")

        # Forward pass (always use soft clustering for inference)
        original_hard_clustering = self.moe_layer.use_hard_clustering
        self.moe_layer.use_hard_clustering = False
        predictions = self(x, training=False)
        self.moe_layer.use_hard_clustering = original_hard_clustering

        # Compute loss
        loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)

        # Update metrics
        self.compiled_metrics.update_state(y, predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results


# if __name__ == "__main__":
#     # Generate dummy dataset with clustered data and labels for training
#     num_train_samples = 8000
#     num_test_samples = 2000
#     feature_dim = 64
#     num_clusters = 32
#
#     # Function to generate dataset with specified parameters
#     def generate_dataset(num_samples, feature_dim, num_clusters, epsilon=0.1):
#         # Compute samples per cluster
#         base_count = num_samples // num_clusters
#         counts = [base_count + (1 if i < num_samples % num_clusters else 0) for i in range(num_clusters)]
#
#         features_list = []
#         labels_list = []
#         cluster_probs_list = []
#         distributions = ['normal', 'uniform', 'gamma', 'poisson']
#
#         for c in range(num_clusters):
#             cluster_count = counts[c]
#             n0 = cluster_count // 2
#             n1 = cluster_count - n0
#             dist = distributions[(2 * c) % len(distributions)]
#             print(f"Cluster {c}: {dist} distribution, {n0} samples 0, {n1} samples 1")
#
#             if dist == 'normal':
#                 features_0 = tf.random.normal([n0, feature_dim], mean=c, stddev=1.0)
#             elif dist == 'uniform':
#                 features_0 = tf.random.uniform([n0, feature_dim], minval=0, maxval=1)
#             elif dist == 'gamma':
#                 features_0 = tf.random.gamma([n0, feature_dim], alpha=2.0, beta=1.0)
#             elif dist == 'poisson':
#                 features_0 = tf.cast(tf.random.poisson([n0, feature_dim], lam=3), tf.float32)
#             else:
#                 features_0 = tf.random.normal([n0, feature_dim], mean=c, stddev=1.0)
#
#             if dist == 'normal':
#                 features_1 = tf.random.normal([n1, feature_dim], mean=c+0.5, stddev=1.5)
#             elif dist == 'uniform':
#                 features_1 = tf.random.uniform([n1, feature_dim], minval=1, maxval=2)
#             elif dist == 'gamma':
#                 features_1 = tf.random.gamma([n1, feature_dim], alpha=5.0, beta=2.0)
#             elif dist == 'poisson':
#                 features_1 = tf.cast(tf.random.poisson([n1, feature_dim], lam=6), tf.float32)
#             else:
#                 features_1 = tf.random.normal([n1, feature_dim], mean=c+0.5, stddev=1.5)
#
#             features_i = tf.concat([features_0, features_1], axis=0)
#             labels_i = tf.concat([tf.zeros([n0, 1], tf.int32), tf.ones([n1, 1], tf.int32)], axis=0)
#             features_list.append(features_i)
#             labels_list.append(labels_i)
#
#             # Generate random cluster probabilities per sample
#             cluster_indices = tf.fill([cluster_count], c)
#             lam_value = tf.maximum(tf.cast(c, tf.float32) + 1.0, 1.0)
#             noise = tf.cast(tf.random.poisson([cluster_count, num_clusters], lam=lam_value), tf.float32)
#             noise = noise + epsilon
#             probs = noise / (tf.reduce_sum(noise, axis=1, keepdims=True))
#             alpha = tf.random.uniform([cluster_count, 1], minval=0.5, maxval=0.8)
#             probs = (1 - alpha) * probs + alpha * tf.one_hot(cluster_indices, num_clusters)
#             probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
#             cluster_probs_list.append(probs)
#
#         features = tf.concat(features_list, axis=0)
#         labels = tf.concat(labels_list, axis=0)
#         cluster_probs = tf.concat(cluster_probs_list, axis=0)
#
#         # Shuffle dataset
#         indices = tf.random.shuffle(tf.range(tf.shape(features)[0]))
#         features = tf.gather(features, indices)
#         labels = tf.gather(labels, indices)
#         cluster_probs = tf.gather(cluster_probs, indices)
#
#         return features, labels, cluster_probs
#
#     # Generate training dataset
#     print("\nGenerating training dataset...")
#     train_features, train_labels, train_cluster_probs = generate_dataset(
#         num_train_samples, feature_dim, num_clusters)
#
#     # Generate separate test dataset
#     print("\nGenerating test dataset...")
#     test_features, test_labels, test_cluster_probs = generate_dataset(
#         num_test_samples, feature_dim, num_clusters, epsilon=1)
#
#     print(f"\nTraining labels min: {tf.reduce_min(train_labels)}, max: {tf.reduce_max(train_labels)}")
#     print(f"Training labels head: {train_labels[:5]}")
#     print(f"Test labels min: {tf.reduce_min(test_labels)}, max: {tf.reduce_max(test_labels)}")
#     print(f"Test labels head: {test_labels[:5]}")
#
#     # Visualize training dataset
#     print("\nVisualizing training dataset...")
#     visualize_dataset_analysis(train_features, train_labels, train_cluster_probs,
#                               raw_dot_plot=False, method='pca', feature_indices=(0, 1))
#
#     # Create training dataset
#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         ((train_features, train_cluster_probs), train_labels)
#     ).shuffle(1000).batch(64)
#
#     # Create test dataset (no need to shuffle extensively)
#     test_dataset = tf.data.Dataset.from_tensor_slices(
#         ((test_features, test_cluster_probs), test_labels)
#     ).batch(64)
#
#     # Print info about test dataset
#     for features, labels in test_dataset.take(1):
#         print(f"\nTest features shape: {features[0].shape}, Test labels shape: {labels.shape}")
#         print(f"Test features head: {features[0][:3]}")
#         print(f"Test labels head: {labels[:3]}")
#
#     # Create and compile model
#     model = EnhancedMoEModel(feature_dim, hidden_dim=8, num_experts=num_clusters, use_hard_clustering=True)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=['accuracy'],
#     )
#
#     # Train model
#     print("\nTraining model...")
#     model.fit(train_dataset, epochs=20)
#
#     # Evaluate on independent test set
#     print("\nEvaluating on independent test set...")
#     eval_results = model.evaluate(test_dataset)
#     print(f"Evaluation results: {eval_results}")
#
#     # Predict and analyze
#     # test_batch = next(iter(test_dataset.take(1)))
#     # predictions = model(test_batch[0], training=False)
#     # print(f"Sample predictions shape: {predictions['prediction'].shape}")
#     # print(f"Gate activations: {tf.reduce_mean(predictions['gates'], axis=0)}")
#
# # Test on a single sample
# for (feat, cluster_prob), label in test_dataset.unbatch().take(1):
#     feat = tf.expand_dims(feat, 0)
#     cluster_prob = tf.expand_dims(cluster_prob, 0)
#     output = model((feat, cluster_prob), training=False)
#     print("Single sample prediction:", output.numpy()[0], "True label:", label.numpy())


# ------------------------------------------------------------------------------- #
# manual implementations
# ----------------------------------------------------------------------------- #
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
                 epsilon=1e-6, gate=True):
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

        self.q = self.add_weight(shape=(heads, input_dim, output_dim),
                                 initializer='random_normal', trainable=True, name=f'q_{name}')
        self.k = self.add_weight(shape=(heads, input_dim, output_dim),
                                 initializer='random_normal', trainable=True, name=f'k_{name}')
        self.v = self.add_weight(shape=(heads, input_dim, output_dim),
                                 initializer='random_normal', trainable=True, name=f'v_{name}')
        if gate:
            self.g = self.add_weight(shape=(heads, input_dim, output_dim),
                                     initializer='random_uniform', trainable=True, name=f'gate_{name}')
        self.norm = layers.LayerNormalization(epsilon=epsilon, name=f'ln_{name}')
        self.norm_out = layers.LayerNormalization(epsilon=epsilon, name=f'ln_out_{name}')
        if resnet:
            self.norm_resnet = layers.LayerNormalization(epsilon=epsilon, name=f'ln_resnet_{name}')
        self.out_w = self.add_weight(shape=(output_dim * heads, output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{name}')
        self.out_b = self.add_weight(shape=(output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{name}')
        self.scale = 1.0 / tf.math.sqrt(tf.cast(output_dim, tf.float32))

    def call(self, x, context=None, mask=None):
        """
        Args:
            x: Tensor of shape (B, N, D)
            context: Optional tensor (B, M, D) for cross-attention
            mask: Optional boolean mask of shape (B, N) or (B, N, 1)
        """
        # Auto-generate padding mask if not provided (based on all-zero tokens)
        if mask is None:
            mask = tf.reduce_sum(tf.abs(x), axis=-1) > 0  # shape: (B, N)
        mask = tf.cast(mask, tf.float32)  # shape: (B, N)

        x_norm = self.norm(x)
        if self.type == 'self':
            q_input = k_input = v_input = x_norm
            mask_k = mask_q = mask
        else:
            assert context is not None, "context is required for cross-attention"
            context_norm = self.norm(context)
            q_input = x_norm
            k_input = v_input = context_norm
            mask_q = tf.cast(tf.reduce_sum(tf.abs(x), axis=-1) > 0, tf.float32)
            mask_k = tf.cast(tf.reduce_sum(tf.abs(context), axis=-1) > 0, tf.float32)

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

    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Create (1, max_len, embed_dim) encoding matrix
        pos = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]  # (max_len, 1)
        i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim)
        angle_rates = 1 / tf.pow(1000.0, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        angle_rads = pos * angle_rates  # (max_len, embed_dim)

        # Apply sin to even indices, cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)  # (max_len, embed_dim)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, max_len, embed_dim)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Optional boolean mask of shape (B, N). True = valid, False = padding
        Returns:
            Tensor with positional encodings added where mask is True.
        """
        seq_len = tf.shape(x)[1]
        pe = self.pos_encoding[:, :seq_len, :]  # (1, N, D)

        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)  # (B, N, 1)
            pe = pe * mask  # zero out positions where mask is 0 (# TODO: check if this is correct)

        return x + pe

# --------------------------------------------------------------------------- #


import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


# --------------------------------------------------------------------------- #
# 1.1 Positional + projection layer for the peptide (21-dim per residue)      #
# --------------------------------------------------------------------------- #
class PeptideProj(layers.Layer):
    """
    Projects peptide vectors (one-hot or 21-dim physicochemical) to embed_dim
    and adds a learned positional embedding.
    """

    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.proj = layers.Dense(embed_dim, use_bias=False, name="peptide_proj")
        self.pos_emb = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=embed_dim,
            name="peptide_pos"
        )

    def call(self, x):
        # x: (batch, S, 21)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project input features
        h = self.proj(x)  # (batch, S, embed_dim)

        # Create position indices
        pos_indices = tf.range(seq_len)  # (S,)
        pos_embeddings = self.pos_emb(pos_indices)  # (S, embed_dim)

        # Add positional embeddings (broadcasting)
        pos_embeddings = tf.expand_dims(pos_embeddings, 0)  # (1, S, embed_dim)
        return h + pos_embeddings  # (batch, S, embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim
        })
        return config


# --------------------------------------------------------------------------- #
# 1.2 Positional + projection layer for the latent (1152-dim per residue)     #
# --------------------------------------------------------------------------- #
class LatentProj(layers.Layer):
    """
    Projects latent vectors (1152-dim) to embed_dim and adds a learned positional
    embedding.
    """

    def __init__(self, max_n_residues, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_n_residues = max_n_residues
        self.proj = layers.Dense(embed_dim, use_bias=False, name="latent_proj")
        self.pos_emb = layers.Embedding(
            input_dim=max_n_residues,
            output_dim=embed_dim,
            name="latent_pos"
        )

    def call(self, x):
        # x: (batch, R, 1152)
        batch_size = tf.shape(x)[0]
        n_residues = tf.shape(x)[1]

        # Project input features
        h = self.proj(x)  # (batch, R, embed_dim)

        # Create position indices
        pos_indices = tf.range(n_residues)  # (R,)
        pos_embeddings = self.pos_emb(pos_indices)  # (R, embed_dim)

        # Add positional embeddings (broadcasting)
        pos_embeddings = tf.expand_dims(pos_embeddings, 0)  # (1, R, embed_dim)
        return h + pos_embeddings  # (batch, R, embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_n_residues": self.max_n_residues,
            "embed_dim": self.embed_dim
        })
        return config


# --------------------------------------------------------------------------- #
# 2.1 Self-attention transformer block                                        #
# --------------------------------------------------------------------------- #
class SelfAttentionBlock(layers.Layer):
    """
    Self-attention block for sequences, followed by FFN + residuals.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            name="self_attn"
        )
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ], name="ffn")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        # x: (B, L, D)
        # Convert mask to proper format for attention if provided
        attention_mask = None
        if mask is not None:
            # mask: (B, L) -> need (B, 1, 1, L) for self-attention
            attention_mask = mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, L)

        # Self-attention
        attn_out = self.attn(
            query=x, key=x, value=x,
            attention_mask=attention_mask,
            training=training
        )
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)  # residual + norm

        # Feed-forward
        ff_out = self.ffn(x)
        ff_out = self.drop2(ff_out, training=training)
        return self.norm2(x + ff_out)  # residual + norm

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


# --------------------------------------------------------------------------- #
# 2.2 Cross-attention transformer block                                       #
# --------------------------------------------------------------------------- #
class CrossAttentionBlock(layers.Layer):
    """
    Cross-attention block with self-attention on queries first, then cross-attention.
    First applies self-attention to queries, then cross-attention with keys/values.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Self-attention for queries
        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            name="self_attn"
        )

        # Cross-attention between queries and keys/values
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            name="cross_attn"
        )

        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ], name="ffn")

        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.norm3 = layers.LayerNormalization(epsilon=1e-6, name="norm3")

        # Dropout layers
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)
        self.drop3 = layers.Dropout(dropout_rate)

    def call(self, queries, keys_values, query_mask=None, key_mask=None, training=False):
        """
        Args:
            queries: (B, L_q, D) - query sequences
            keys_values: (B, L_kv, D) - key/value sequences
            query_mask: (B, L_q) - mask for queries
            key_mask: (B, L_kv) - mask for keys/values
            training: bool
        """
        # Convert masks to proper format for attention if provided
        query_attention_mask = None
        if query_mask is not None:
            # query_mask: (B, L_q) -> need (B, 1, 1, L_q) for self-attention
            query_attention_mask = query_mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, L_q)

        key_attention_mask = None
        if key_mask is not None:
            # key_mask: (B, L_kv) -> need (B, 1, 1, L_kv) for cross-attention
            key_attention_mask = key_mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, L_kv)

        # Step 1: Self-attention on queries
        self_attn_out = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
            attention_mask=query_attention_mask,
            training=training
        )
        self_attn_out = self.drop1(self_attn_out, training=training)
        queries_refined = self.norm1(queries + self_attn_out)  # residual + norm

        # Step 2: Cross-attention between refined queries and keys/values
        cross_attn_out = self.cross_attn(
            query=queries_refined,
            key=keys_values,
            value=keys_values,
            attention_mask=key_attention_mask,
            training=training
        )
        cross_attn_out = self.drop2(cross_attn_out, training=training)
        x = self.norm2(queries_refined + cross_attn_out)  # residual connection

        # Step 3: Feed-forward network
        ff_out = self.ffn(x)
        ff_out = self.drop3(ff_out, training=training)
        return self.norm3(x + ff_out)  # residual + norm

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


# --------------------------------------------------------------------------- #
# 3. Build the complete classifier                                            #
# --------------------------------------------------------------------------- #
def build_classifier(max_seq_len=50,
                     max_n_residues=500,
                     n_blocks=4,
                     embed_dim=256,
                     num_heads=8,
                     ff_dim=512,
                     dropout_rate=0.01):
    """
    Build peptide-latent interaction classifier.

    Args:
        max_seq_len: Maximum peptide sequence length
        max_n_residues: Maximum number of protein residues
        n_blocks: Number of transformer blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """

    # --- Inputs -------------------------------------------------------------
    peptide_in = layers.Input(shape=(None, 21), name="peptide")  # (B, S, 21)
    latent_in = layers.Input(shape=(None, 1152), name="latent_raw")  # (B, R, 1152)

    # --- Create attention masks ---------------------------------------------
    # Peptide mask: True where peptide has content (non-zero vectors)
    pep_mask = tf.reduce_any(tf.abs(peptide_in) > 1e-6, axis=-1)  # (B, S)

    # Latent mask: True where latent has content (non-zero vectors)
    latent_mask = tf.reduce_any(tf.abs(latent_in) > 1e-6, axis=-1)  # (B, R)

    # --- Projections --------------------------------------------------------
    pep_proj = PeptideProj(max_seq_len, embed_dim, name="peptide_projection")(peptide_in)
    latent_proj = LatentProj(max_n_residues, embed_dim, name="latent_projection")(latent_in)

    # --- Self-attention blocks for latent representation -------------------
    latent_embed = latent_proj
    for i in range(n_blocks):
        latent_embed = SelfAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"latent_self_attn_block_{i + 1}"
        )(latent_embed, mask=latent_mask)

    # --- Cross-attention fusion ---------------------------------------------
    # Latent queries attend to peptide keys/values
    fused = latent_embed
    for i in range(n_blocks):
        fused = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name=f"cross_attn_block_{i + 1}"
        )(queries=fused, keys_values=pep_proj, query_mask=latent_mask, key_mask=pep_mask)


    # --- Aggregation and prediction head -----------------------------------
    # Global average pooling with masking
    latent_mask_expanded = tf.expand_dims(tf.cast(latent_mask, tf.float32), -1)  # (B, R, 1)
    masked_fused = fused * latent_mask_expanded  # Zero out padded positions

    # Compute mean only over valid positions
    pooled = tf.reduce_sum(masked_fused, axis=1)  # (B, D)
    valid_lengths = tf.reduce_sum(latent_mask_expanded, axis=1)  # (B, 1)
    pooled = pooled / (valid_lengths + 1e-8)  # Average over valid positions

    # Simpler classification head
    output = layers.Dense(1, activation="sigmoid", name="output")(pooled)
    # # Final prediction layers
    # x = layers.Dense(embed_dim, activation="relu", name="pred_hidden")(pooled)
    # x = layers.Dropout(dropout_rate)(x)
    # x = layers.Dense(embed_dim // 2, activation="relu", name="pred_hidden2")(x)
    # x = layers.Dropout(dropout_rate)(x)
    # output = layers.Dense(1, activation="softmax", name="output")(x)

    # --- Build and compile model --------------------------------------------
    model = Model(
        inputs=[peptide_in, latent_in],
        outputs=output,
        name="peptide_latent_classifier"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy", "AUC"]
    )

    return model


# ----------------------------------------------------------------------------- #
# # 4. Utility function to test the model                                       #
# # --------------------------------------------------------------------------- #
# def test_model():
#     """Test the model with dummy data to ensure it works."""
#
#     # Create model
#     model = build_classifier(
#         max_seq_len=20,
#         max_n_residues=100,
#         n_blocks=2,
#         embed_dim=128,
#         num_heads=4,
#         ff_dim=256,
#         dropout_rate=0.1
#     )
#
#     # Print model summary
#     print("Model Summary:")
#     model.summary()
#
#     # Create dummy data
#     batch_size = 4
#     seq_len = 15
#     n_residues = 80
#
#     # Dummy peptide data (one-hot encoded)
#     peptide_data = tf.random.uniform((batch_size, seq_len, 21), maxval=2, dtype=tf.int32)
#     peptide_data = tf.cast(peptide_data, tf.float32)
#
#     # Dummy latent data
#     latent_data = tf.random.normal((batch_size, n_residues, 1152))
#
#     # Test forward pass
#     print("\nTesting forward pass...")
#     predictions = model([peptide_data, latent_data])
#     print(f"Output shape: {predictions.shape}")
#     print(f"Sample predictions: {predictions[:3].numpy().flatten()}")
#
#     # Test with different sequence lengths
#     print("\nTesting with variable sequence lengths...")
#     peptide_data2 = tf.random.uniform((batch_size, 10, 21), maxval=2, dtype=tf.int32)
#     peptide_data2 = tf.cast(peptide_data2, tf.float32)
#     latent_data2 = tf.random.normal((batch_size, 60, 1152))
#
#     predictions2 = model([peptide_data2, latent_data2])
#     print(f"Output shape with different lengths: {predictions2.shape}")
#
#     print("\nModel test completed successfully!")
#
#     return model
#
#
# if __name__ == "__main__":
#     # Test the model
#     test_model()


# # --------------------------------------------------------------------------- #
# # 4. Utility function to test the model                                      #
# # --------------------------------------------------------------------------- #
# def test_model():
#     """Test the model with dummy data to ensure it works."""
#
#     # Create model
#     model = build_classifier(
#         max_seq_len=20,
#         max_n_residues=100,
#         n_blocks=2,
#         embed_dim=128,
#         num_heads=4,
#         ff_dim=256,
#         dropout_rate=0.1
#     )
#
#     # Print model summary
#     print("Model Summary:")
#     model.summary()
#
#     # Create dummy data
#     batch_size = 4
#     seq_len = 15
#     n_residues = 80
#
#     # Dummy peptide data (one-hot encoded)
#     peptide_data = tf.random.uniform((batch_size, seq_len, 21), maxval=2, dtype=tf.int32)
#     peptide_data = tf.cast(peptide_data, tf.float32)
#
#     # Dummy latent data
#     latent_data = tf.random.normal((batch_size, n_residues, 1152))
#
#     # Test forward pass
#     print("\nTesting forward pass...")
#     predictions = model([peptide_data, latent_data])
#     print(f"Output shape: {predictions.shape}")
#     print(f"Sample predictions: {predictions[:3].numpy().flatten()}")
#
#     # Test with different sequence lengths
#     print("\nTesting with variable sequence lengths...")
#     peptide_data2 = tf.random.uniform((batch_size, 10, 21), maxval=2, dtype=tf.int32)
#     peptide_data2 = tf.cast(peptide_data2, tf.float32)
#     latent_data2 = tf.random.normal((batch_size, 60, 1152))
#
#     predictions2 = model([peptide_data2, latent_data2])
#     print(f"Output shape with different lengths: {predictions2.shape}")
#
#     print("\nModel test completed successfully!")
#
#     return model
#
#
# if __name__ == "__main__":
#     # Test the model
#     test_model()

# --------------------------------------------------------------------------- #
# 4. Demo: instantiate and inspect                                           #
# --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     SEQ_LEN = 15       # typical peptide length (adjust to your data)
#
#     model = build_classifier(max_seq_len=SEQ_LEN)
#     model.summary()
#
#     # Training example (dummy):
#     peptide_batch = tf.random.uniform((320, SEQ_LEN, 21))
#     latent_batch  = tf.random.uniform((320, 36, 1152))
#     labels        = tf.random.uniform((320, 1), maxval=2, dtype=tf.int32)
#     model.fit([peptide_batch, latent_batch], labels, epochs=100)


## Barcode peptides ##
# a model that creates a barcode for peptides by taking 9mer windows and returning a 1D vector that represents