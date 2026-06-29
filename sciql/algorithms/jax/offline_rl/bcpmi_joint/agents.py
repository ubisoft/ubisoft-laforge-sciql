import flax
import jax
import optax
import functools
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
import json

from pydantic import BaseModel
from flax.training.train_state import TrainState
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from sciql.core.data import Sampler
from sciql.core.agent import Agent
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.bcpmi_joint.samplers.samplers import BCPMI_JOINT_Batch, Dummy_Sampler
from sciql.algorithms.jax.offline_rl.bcpmi_joint.networks import GLCDiscreteActor, GLCActor, GLCDiscreteCritic, GLCValue
from sciql.algorithms.jax.offline_rl.bcpmi_joint.encoders import gc_encoder_modules, GCEncoder, lc_encoder_modules

def update_by_loss_grad(train_state: TrainState, loss_fn: Callable) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss

class BCPMI_JOINT_TrainState(NamedTuple):
    mine: TrainState
    actor: TrainState

class BCPMI_JOINT_Config(BaseModel):
    """
    A Pydantic configuration class for the Behavior Cloning agent.
    Provides type safety, validation, and clear defaults.
    """
    class Config:
        frozen = True  # Makes instances immutable and hashable.

    # Seeding
    seed: int = 0

    # Architecture
    hidden_dims: Tuple[int, int] = (256, 256)
    label_embed_dim: int = 16
    gc_encoder: Optional[str] = None
    lc_encoder: Optional[str] = 'embedding'
    discrete: bool = False
    action_dim: Optional[int] = None

    # Algorithm
    normalize_state: bool = False

    # Optimization
    actor_lr: float = 3e-4
    mine_lr: float = 3e-4
    batch_size: int = 256
    opt_decay_schedule: bool = True
    policy_max_steps: int = int(1e6)
    policy_n_jitted_updates: int = 1
    mine_n_jitted_updates: int = 1

class BCPMI_JOINT_Agent(flax.struct.PyTreeNode, Agent):

    rng: jax.random.PRNGKey
    eval_rng: jax.random.PRNGKey
    eval_labels: Any = nonpytree_field()
    config: BCPMI_JOINT_Config = nonpytree_field()
    train_state: BCPMI_JOINT_TrainState
    obs_mean: Any = nonpytree_field()
    obs_std: Any = nonpytree_field()
    labels_numbers: Any = nonpytree_field()
    batch_structure: Any = nonpytree_field()
    
    ##################
    # Initialization #
    ##################
    @classmethod
    def create(
        cls,
        sampler: Sampler,
        cfg: Optional[dict] = None,
        config: Optional[BCPMI_JOINT_Config] = None,
    ):
        # Get config from cfg or use the provided config object
        if config is None:
            if cfg is None: cfg = {}
            config = BCPMI_JOINT_Config(**cfg)

        # Set jax seed
        rng = jax.random.PRNGKey(config.seed)
        rng, eval_rng, batch_rng, mine_rng, actor_rng = jax.random.split(rng, 5)

        # Get data from batch
        example_batch: BCPMI_JOINT_Batch = sampler.sample_batch(1, batch_rng)
        observations = example_batch.observations
        actions = example_batch.actions
        labels = example_batch.joint_labels
        batch_structure = jax.tree.map(lambda x: (x.shape, str(x.dtype)), example_batch)

        # Get action_dim
        if config.discrete:
            action_dim = config.action_dim if config.action_dim is not None else actions.max() + 1
        else:
            action_dim = config.action_dim if config.action_dim is not None else actions.shape[-1]
        
        # Define gc_encoders
        gc_encoders = dict()
        if config.gc_encoder is not None:
            gc_encoder_module = gc_encoder_modules[config.gc_encoder]
            gc_encoders['mine'] = GCEncoder(state_encoder=gc_encoder_module())
            gc_encoders['actor'] = GCEncoder(state_encoder=gc_encoder_module())
        
        # Define lc_encoders
        lc_encoders = dict()
        if config.lc_encoder is not None:
            lc_encoder_module = lc_encoder_modules[config.lc_encoder]
            lc_encoders['mine'] = lc_encoder_module(labels_numbers=sampler.labels_numbers, label_embed_dim=config.label_embed_dim)
            lc_encoders['actor'] = lc_encoder_module(labels_numbers=sampler.labels_numbers, label_embed_dim=config.label_embed_dim)

        # Initlialize modules

        # MINE train state
        if config.discrete:
            mine_model = GLCDiscreteCritic(
                hidden_dims=config.hidden_dims,
                gc_encoder=gc_encoders.get('mine'),
                lc_encoder=lc_encoders.get('mine'),
                action_dim=action_dim,
            )
        else:
            mine_model = GLCValue(
                hidden_dims=config.hidden_dims,
                gc_encoder=gc_encoders.get('mine'),
                lc_encoder=lc_encoders.get('mine'),
            )
        mine = TrainState.create(
            apply_fn=mine_model.apply,
            params=mine_model.init(mine_rng, observations, actions=actions, labels=labels),
            tx=optax.adam(learning_rate=config.mine_lr),
        )

        # Actor train state
        if config.discrete:
            actor_model = GLCDiscreteActor(
                hidden_dims=config.hidden_dims,
                action_dim=action_dim,
                gc_encoder=gc_encoders.get('actor'),
                lc_encoder=lc_encoders.get('actor'),
            )
        else:
            actor_model = GLCActor(
                hidden_dims=config.hidden_dims,
                action_dim=action_dim,
                gc_encoder=gc_encoders.get('actor'),
                lc_encoder=lc_encoders.get('actor'),
            )
        if config.opt_decay_schedule:
            schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.policy_max_steps)
            actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=config.actor_lr)
        actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_rng, observations, labels=labels),
            tx=actor_tx,
        )

        # Set train state
        train_state = BCPMI_JOINT_TrainState(
            mine=mine,
            actor=actor
        )

        return cls(
            rng=rng,
            eval_rng=eval_rng,
            eval_labels=labels,
            config=config,
            train_state=train_state,
            obs_mean=sampler.obs_mean,
            obs_std=sampler.obs_std,
            labels_numbers=sampler.labels_numbers,
            batch_structure=batch_structure
        )

    ####################
    # Saving & Loading #
    ####################
    def save(self, path: str):
        """
        Saves the agent by converting the data structure and config to a dict for
        easier and more robust serialization.
        """
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, self)

        # 1. Get the structure of the data part of the batch.
        data_structure = self.batch_structure

        # 2. Convert the Data NamedTuple to an ordered dictionary.
        data_dict = data_structure._asdict()

        # 3. Use a simple dictionary comprehension. This is much more readable!
        serializable_structure = {
            key: {'shape': list(val[0]), 'dtype': val[1]}
            for key, val in data_dict.items()
        }

        # 4. Add attributess
        metadata = {
            'data_structure': serializable_structure,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'labels_numbers': self.labels_numbers,
            'config': self.config.model_dump()
        }

        # 5. Save attributes
        metadata_path = os.path.join(path, 'agent_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Agent and metadata successfully saved to: {path}")
        return self

    @classmethod
    def load(cls, path: str) -> "BCPMI_JOINT_Agent":
        """
        Loads an agent by first reading the explicit metadata file, then restoring.
        """
        metadata_path = os.path.join(path, 'agent_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Required 'agent_metadata.json' not found in checkpoint: {path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 1. Load the dictionary containing the serialized structure.
        serializable_dict = metadata['data_structure']

        # 2. Use a dictionary comprehension to reconstruct the (shape, dtype) tuples.
        reconstructed_dict = {
            key: (tuple(val['shape']), val['dtype'])
            for key, val in serializable_dict.items()
        }

        # 3. Unpack the dictionary directly into the Data NamedTuple constructor.
        data_structure = BCPMI_JOINT_Batch(**reconstructed_dict)

        # 4. Retrieve other sampler's attributes
        obs_mean = float(metadata['obs_mean'])
        obs_std = float(metadata['obs_std'])
        labels_numbers = tuple(metadata['labels_numbers'])

        # 5. Create the config object from the loaded dictionary
        config = BCPMI_JOINT_Config(**metadata['config'])

        # 6. Create the abstract agent.
        abstract_agent = jax.eval_shape(
            lambda: cls.create_abstract(
                data_structure=data_structure,
                obs_mean=obs_mean,
                obs_std=obs_std,
                labels_numbers=labels_numbers,
                config=config  # Pass the loaded config
            )
        )

        # 7. Restore the agent state.
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(path, item=abstract_agent)

    @classmethod
    def create_abstract(cls, data_structure: Any, obs_mean: float, obs_std: float, labels_numbers: tuple, config: BCPMI_JOINT_Config):
        """Internal helper that creates a dummy agent from data metadata."""
        dummy_sampler = Dummy_Sampler(
            data_structure=data_structure,
            obs_mean=obs_mean,
            obs_std=obs_std,
            labels_numbers=labels_numbers
        )
        return cls.create(sampler=dummy_sampler, config=config)
    
    #################
    # MINE Training #
    #################
    def update_mine_one_step(
        self,
        batch: BCPMI_JOINT_Batch,
        train_state_mine: TrainState
    ) -> tuple[TrainState, dict]:
        """
        One mine step
        """
        def mine_loss_fn(params: flax.core.FrozenDict) -> jnp.ndarray:
            t_joint = train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.joint_labels)
            t_marginal = train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.marginal_labels)
            mi_estimate = jnp.mean(t_joint) - jnp.log(jnp.mean(jnp.exp(t_marginal)) + 1e-8)
            return -mi_estimate

        new_mine_train_state, loss = update_by_loss_grad(train_state_mine, mine_loss_fn)
        return new_mine_train_state, loss

    @jax.jit
    def update_mine_n_steps(self, sampler: Sampler) -> tuple["BCPMI_JOINT_Agent", dict]:
        """
        Loop on the jitted updates
        """
        rng = self.rng
        new_mine_train_state = self.train_state.mine
        total_loss = 0.0

        for _ in range(self.config.mine_n_jitted_updates):
            rng, batch_rng = jax.random.split(rng)
            batch = sampler.sample_batch(self.config.batch_size, batch_rng)
            new_mine_train_state, step_loss = self.update_mine_one_step(batch, new_mine_train_state)
            total_loss += step_loss

        mean_loss = total_loss / self.config.mine_n_jitted_updates
        infos = {f'mine_loss': mean_loss}

        new_train_state = self.train_state._replace(mine=new_mine_train_state)
        return self.replace(train_state=new_train_state, rng=rng), infos
    
    ###################
    # Policy Training #
    ###################
    def update_actor(
        self,
        train_state: BCPMI_JOINT_TrainState, # Actor needs access to all critics/values
        batch: BCPMI_JOINT_Batch,
    ) -> Tuple[TrainState, Dict]:

        # Compute PMI scores
        pmi = train_state.mine.apply_fn(train_state.mine.params, batch.observations, actions=batch.actions, labels=batch.joint_labels)
        weights = jnp.exp(pmi)

        # For now, we don't add the baseline as it provokes instabilities during training.
        def actor_loss_fn(actor_params: flax.core.FrozenDict) -> jnp.ndarray:
            dist = train_state.actor.apply_fn(actor_params, batch.observations, labels=batch.joint_labels)
            log_probs = dist.log_prob(batch.actions)
            actor_loss = -(weights * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        return train_state._replace(actor=new_actor), actor_loss

    @jax.jit
    def update_train_state(self, batch, train_state):
        train_state, actor_loss = self.update_actor(train_state, batch)
        return train_state, {"actor_loss": actor_loss}

    def update_policy(self, sampler: Sampler):

        # Set rng and train_state
        rng = self.rng
        train_state = self.train_state
        
        # Train one step
        rng, batch_rng = jax.random.split(rng)
        batch = sampler.sample_batch(self.config.batch_size, batch_rng)
        train_state, step_losses = self.update_train_state(batch, train_state)

        return self.replace(train_state=train_state, rng=rng), step_losses

    @jax.jit
    def update_policy_n_steps(self, sampler: Sampler) -> Tuple["BCPMI_JOINT_Agent", Dict]:
        """
        Main training function for the policy. Contains the outer Python loop over training steps.
        """
        # Set rng and train_state
        rng = self.rng
        train_state = self.train_state
        
        # Create accumulator
        total_losses = jax.tree_util.tree_map(lambda x: x, {'actor_loss': 0.0})

        # Train over all jitted steps
        for _ in range(self.config.policy_n_jitted_updates):
            rng, batch_rng = jax.random.split(rng)
            batch = sampler.sample_batch(self.config.batch_size, batch_rng)
            train_state, step_losses = self.update_train_state(batch, train_state)
            total_losses = jax.tree_util.tree_map(lambda x, y: x + y, total_losses, step_losses)
    
        # Calculate the mean loss over all the steps
        mean_losses = jax.tree_util.tree_map(lambda x: x / self.config.policy_n_jitted_updates, total_losses)

        return self.replace(train_state=train_state, rng=rng), mean_losses
    
    #############
    # Inference #
    #############
    @jax.jit
    def act(
        self,
        observations: np.ndarray,
        temperature: float = 0.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
    ) -> jnp.ndarray:
        rng, labels_rng, policy_rng = jax.random.split(self.eval_rng, 3)
        labels_rngs = jax.random.split(labels_rng, len(self.labels_numbers))

        observations = observations['full']
        if self.config.normalize_state:
            observations = (observations - self.obs_mean) / (self.obs_std + 1e-5)
        observations = jnp.array([observations], dtype=jnp.float32)

        actions = self.train_state.actor.apply_fn(self.train_state.actor.params, observations, labels=self.eval_labels, temperature=temperature).sample(seed=policy_rng)
        if self.config.discrete:
            actions = actions.astype(jnp.int32)
        else:
            actions = jnp.clip(actions, -max_action, max_action).astype(jnp.float32)
            if actions.ndim > 1: 
                actions = jnp.squeeze(actions, axis=0)
        
        return self.replace(eval_rng=rng), {'full': actions}
    
    def reset(
        self,
        seed: int,
        labels: np.ndarray
    ):  
        print(f'Resetting agent with evaluation labels: {labels}')
        rng = jax.random.PRNGKey(seed)
        return self.replace(eval_rng=rng, eval_labels=jnp.array([labels]))

    #########
    # Utils #
    #########
    def set_train_mode(self):
        return self
    
    def set_eval_mode(self):
        return self

    def to(self, device: str) -> "BCPMI_JOINT_Agent":
        lower_device = device.lower()
        try:
            if lower_device == 'cpu':
                target_device = jax.devices('cpu')[0]
            elif lower_device in ('gpu', 'cuda'):
                # Check if a GPU is available first to provide a clear error
                gpu_devices = jax.devices('gpu')
                if not gpu_devices:
                    raise ValueError("Requested device 'gpu'/'cuda', but no JAX-compatible GPU is available.")
                target_device = gpu_devices[0]
            else:
                raise ValueError(f"Unsupported device string '{device}'. Please use 'cpu' or 'gpu'.")
        except IndexError:
            # This is a fallback error in case jax.devices() returns an empty list for an expected device
            raise RuntimeError(f"Could not retrieve a device object for backend '{lower_device}'.")
        return jax.device_put(self, target_device)


