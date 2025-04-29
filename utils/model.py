# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class SCQ_layer(layers.Layer):
    def __init__(self, num_embed, dim_embed, lambda_reg=1.0, proj_iter=10,
                 discrete_loss=False, beta_loss=0.25, reset_dead_codes=False,
                 usage_threshold=1e-3, reset_interval=2, **kwargs):
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
                 scq_params, initial_codebook, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim  # e.g., (1024,)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_beta = commitment_beta

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

    def call(self, inputs, training=False):
        # Encoder: Compress input to embedding space
        x = self.encoder(inputs)  # (batch_size, embedding_dim)
        x = tf.expand_dims(x, axis=1)  # (batch_size, 1, embedding_dim)

        # SCQ: Quantize the embedding
        # This now returns one-hot encodings instead of soft probabilities
        Zq, out_P_proj, vq_loss, perplexity = self.scq_layer(x)

        # Decoder: Reconstruct from quantized embedding
        y = tf.squeeze(Zq, axis=1)  # (batch_size, embedding_dim)
        output = self.decoder(y)  # (batch_size, 1024)

        return output, Zq, out_P_proj, vq_loss, perplexity

    # # Method to get just the latent sequence and one-hot encodings for inference
    # def encode(self, inputs):
    #     """Encode inputs to latent sequence and one-hot cluster assignments."""
    #     x = self.encoder(inputs)  # (batch_size, embedding_dim)
    #     x = tf.expand_dims(x, axis=1)  # (batch_size, 1, embedding_dim)
    #
    #     # Get quantized embedding and one-hot encodings
    #     quantized, one_hot_encodings, _, _ = self.scq_layer(x)
    #
    #     # Return quantized latent sequence and one-hot encodings
    #     return tf.squeeze(quantized, axis=1), tf.squeeze(one_hot_encodings, axis=1)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker,
            self.perplexity_tracker
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
            y = data[1] if len(data) > 1 else data[0]
        else:
            x = y = data

        with tf.GradientTape() as tape:
            reconstruction, Zq, _, vq_loss, perplexity = self(x, training=True)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(tf.math.squared_difference(y, reconstruction))

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
            x = y = data

        reconstruction, quantized, _, vq_loss, perplexity = self(x, training=False)
        recon_loss = tf.reduce_mean(tf.math.squared_difference(y, reconstruction))
        total_loss = recon_loss + vq_loss + self.commitment_beta * tf.reduce_mean(
            tf.math.squared_difference(x, reconstruction))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        self.perplexity_tracker.update_state(perplexity)

        return {m.name: m.result() for m in self.metrics}


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
    """A binary prediction expert."""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = layers.Dense(hidden_dim, activation='relu')
        self.fc2 = layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.nn.sigmoid(x)

class MixtureOfExperts(layers.Layer):
    """Mixture of Experts layer using provided cluster probabilities."""
    def __init__(self, input_dim, hidden_dim, num_experts, use_provided_gates=True):
        super().__init__()
        self.num_experts = num_experts
        self.use_provided_gates = use_provided_gates
        self.experts = [Expert(input_dim, hidden_dim) for _ in range(num_experts)]

    def call(self, inputs, training=False):
        if self.use_provided_gates:
            if isinstance(inputs, tuple) and len(inputs) == 2:
                x, cluster_probs = inputs
                gates = tf.nn.softmax(cluster_probs, axis=-1)
            else:
                raise ValueError("Inputs must be a tuple of (features, cluster_probs) when use_provided_gates=True")

        experts_used = tf.reduce_sum(tf.cast(gates > 0.01, tf.float32), axis=1)
        expert_influence = tf.reduce_sum(gates, axis=0)
        expert_activation_count = tf.reduce_sum(tf.cast(gates > 0.01, tf.float32), axis=0)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = tf.cond(
                tf.greater(tf.shape(expert_inputs[i])[0], 0),
                lambda: self.experts[i](expert_inputs[i]),
                lambda: tf.zeros((0, 1), dtype=tf.float32)
            )
            expert_outputs.append(expert_output)

        y = dispatcher.combine(expert_outputs, multiply_by_gates=True)

        return {
            'prediction': y,
            'gates': gates,
            'experts_used': experts_used,
            'expert_influence': expert_influence,
            'expert_activation_count': expert_activation_count
        }

class MoEModel(tf.keras.Model):
    """Model wrapping the MoE layer."""
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.moe_layer = MixtureOfExperts(input_dim, hidden_dim, num_experts)

    def call(self, inputs, training=False):
        moe_outputs = self.moe_layer(inputs, training=training)
        if training:
            return moe_outputs['prediction']
        return moe_outputs

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            moe_outputs = self.moe_layer(x, training=True)
            y_pred = moe_outputs['prediction']
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        y = tf.squeeze(y) if len(y.shape) > 1 else y
        y_pred = tf.squeeze(y_pred) if len(y_pred.shape) > 1 else y_pred
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

# Example usage
if __name__ == "__main__":
    # Generate dummy dataset
    num_samples = 10000
    feature_dim = 128
    num_clusters = 32

    features = tf.random.normal([num_samples, feature_dim])
    labels = tf.random.uniform([num_samples, 1], minval=0, maxval=2, dtype=tf.int32)
    random_logits = tf.random.normal([num_samples, num_clusters])
    cluster_probs = tf.nn.softmax(random_logits, axis=-1)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(((features, cluster_probs), labels))
    dataset = dataset.batch(32)

    # Split into train and test
    train_size = int(0.8 * num_samples)
    train_dataset = dataset.take(train_size // 32)
    test_dataset = dataset.skip(train_size // 32)

    # Create and compile model
    model = MoEModel(input_dim=feature_dim, hidden_dim=256, num_experts=num_clusters)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Train
    print("Training model...")
    model.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=1)

    # Evaluate
    print("\nEvaluating model...")
    results = model.evaluate(test_dataset, return_dict=True)
    print(f"Test loss: {results['loss']:.4f}, Test accuracy: {results['binary_accuracy']:.4f}")

    # Predict and analyze
    sample_features, sample_labels = next(iter(test_dataset))
    predictions = model(sample_features, training=False)

    print(f"\nSample predictions shape: {predictions['prediction'].shape}")
    print(f"First 5 predictions:\n{predictions['prediction'][:5].numpy()}")
    print(f"First 5 actual labels:\n{sample_labels[:5].numpy()}")
    print(f"Average experts used per sample: {tf.reduce_mean(predictions['experts_used']):.2f}")
    print(f"Expert influence distribution: {predictions['expert_influence'].numpy()}")

    top_experts = tf.argsort(predictions['expert_influence'], direction='DESCENDING')[:5]
    print(f"Top 5 most influential experts: {top_experts.numpy()}")