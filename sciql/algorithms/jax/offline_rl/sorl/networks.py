# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple
import distrax
import flax.linen as nn
import jax.numpy as jnp

def orthogonal_init(scale: Optional[float] = jnp.sqrt(2)):
    """Default kernel initialer from JAX_CORL."""
    return nn.initializers.orthogonal(scale)

def variance_scaling_init(scale=1.0):
    """Default kernel initializer from OGBench."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )

class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())

class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
        layer_norm_after: Wether to apply the layer normalization before (JAX_CORL) or after (OGBench) the activation.
    """

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = orthogonal_init()
    layer_norm: bool = False
    layer_norm_after: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims):
                if self.layer_norm and not self.layer_norm_after:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.layer_norm and self.layer_norm_after:
                    x = nn.LayerNorm()(x)
            elif self.activate_final:
                if self.layer_norm and not self.layer_norm_after:
                    x = nn.LayerNorm()(x)
                x = self.output_activation(x)
                if self.layer_norm and self.layer_norm_after:
                    x = nn.LayerNorm()(x)
        return x
    
class GLCActor(nn.Module):
    """Goal-label-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        kernel_init: Kernel initializer. It is orthogonal_init() for JAX_CORL and variance_scaling_init(1e-2) for OGBench.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        lc_encoder: Optional label encoder to encode the labels.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = 2.0
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = False
    kernel_init: Any = orthogonal_init()
    gc_encoder: nn.Module = None
    lc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=self.kernel_init)
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=self.kernel_init)
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        labels=None,
        labels_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            labels: Labels (optional)
            labels_encoded: Whether the labels are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if labels is not None:
            if self.lc_encoder is not None:
                inputs.append(self.lc_encoder(labels, labels_encoded=labels_encoded))
            else:
                inputs.append(labels)
        inputs = jnp.concatenate(inputs, axis=-1)

        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution

class GLCDiscreteActor(nn.Module):
    """Goal-label-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        kernel_init: Kernel initializer. It is orthogonal_init() for JAX_CORL and variance_scaling_init(1e-2) for OGBench.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        lc_encoder: Optional label encoder to encode the labels.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    kernel_init: Any = orthogonal_init()
    gc_encoder: nn.Module = None
    lc_encoder: nn.Module = None

    def setup(self):        
        # Actor
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=self.kernel_init)

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        labels=None,
        labels_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if labels is not None:
            if self.lc_encoder is not None:
                inputs.append(self.lc_encoder(labels, labels_encoded=labels_encoded))
            else:
                inputs.append(labels)
        inputs = jnp.concatenate(inputs, axis=-1)
        
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution

class GLCValue(nn.Module):
    """Goal-label-conditioned lavel value/critic function.

    This module can be used for both:
        - value V(s)
        - critic Q(s, a)
        - label conditioned value V(s, z)
        - label conditioned critic Q(s, a, z)
        - goal-label-conditioned value V(s, g, z)
        - goal-label-conditioned critic Q(s, a, g, z) 
    functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        lc_encoder: Optional label encoder to encode the labels.
    """

    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    squeeze_output: bool = True
    activate_final : bool = False
    layer_norm: bool = False
    ensemble: bool = False
    gc_encoder: nn.Module = None
    lc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module(
            (*self.hidden_dims, self.output_dim),
            activations=self.activations,
            layer_norm=self.layer_norm,
            activate_final=self.activate_final,
            output_activation=self.output_activation
        )
        self.value_net = value_net

    def __call__(self, observations, actions=None, goals=None, labels=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
            labels: Labels (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        if labels is not None:
            if self.lc_encoder is not None:
                inputs.append(self.lc_encoder(labels))
            else:
                inputs.append(labels)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.squeeze_output:
            v = v.squeeze(-1)

        return v

class GLCDiscreteCritic(GLCValue):
    """Goal-label-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, actions=None, goals=None, labels=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, actions, goals, labels)