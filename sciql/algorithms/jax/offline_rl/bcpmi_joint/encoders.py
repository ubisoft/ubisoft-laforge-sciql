import functools
from typing import Sequence, Optional

import flax.linen as nn
import jax.numpy as jnp

from sciql.algorithms.jax.offline_rl.bc.networks import MLP

class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

class Mnih2015Encoder(nn.Module):
    """A Flax encoder that replicates the CNN from (Mnih et al., 2015).

    This module is designed to be used as a `gc_encoder` in the GCDiscreteActor.
    It takes image observations and outputs a flat feature vector.

    Note: The original paper's architecture implicitly assumes channels-first (NCHW)
    input. This implementation works with channels-last (NHWC) which is standard
    for Flax/JAX.
    """

    @nn.compact
    def __call__(self, x):
        """
        CNN head similar to one used in Mnih 2015
       (Human-level control through deep reinforcement learning, Mnih 2015)

        Args:
            observations: A batch of image observations, expected in NHWC format.
            goals: Ignored. Included for signature compatibility with GCDiscreteActor.
            goal_encoded: Ignored. For compatibility.

        Returns:
            A flat feature vector for each image in the batch.
        """
        # Ensure input is float and normalized, making the model self-contained.
        # This is good practice even if the sampler already does it.
        x = x.astype(jnp.float32) / 255.0

        # Conv Layer 1: 32 filters, 8x8 kernel, 4x4 stride
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), padding='VALID')(x)
        x = nn.relu(x)

        # Conv Layer 2: 64 filters, 4x4 kernel, 2x2 stride
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)

        # Conv Layer 3: 64 filters, 3x3 kernel, 1x1 stride
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)

        # Flatten the output for the MLP
        # The shape will be (batch_size, flattened_features)
        x = x.reshape((x.shape[0], -1))

        return x

class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


gc_encoder_modules = {
    'mnih2015': Mnih2015Encoder,
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}

class MultiLabelEmbeddingEncoder(nn.Module):
    """ Multi-label encoder.
    """
    labels_numbers: Sequence[int]
    label_embed_dim: int

    def setup(self):
        self.embeddings_layers = [
            nn.Embed(num_embeddings=num_categories, features=self.label_embed_dim)
            for num_categories in self.labels_numbers
        ]
    
    def __call__(
        self,
        labels,
        labels_encoded=False
    ):
        
        if labels_encoded:
            return labels
        
        if labels.ndim == 1:
            labels = jnp.expand_dims(labels, axis=-1)

        all_embeddings = []
        for i, embedding_layer in enumerate(self.embeddings_layers):
            label_slice = labels[:, i]
            embedding = embedding_layer(label_slice)
            all_embeddings.append(embedding)
        combined_embeddings = jnp.concatenate(all_embeddings, axis=-1)
        return combined_embeddings

lc_encoder_modules = {
    'embedding': MultiLabelEmbeddingEncoder,
}
