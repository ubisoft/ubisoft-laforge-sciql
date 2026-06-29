import flax
import jax
import optax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
import json

import flax.linen as nn
from pydantic import BaseModel
from flax.training.train_state import TrainState
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from sciql.core.data import Sampler
from sciql.core.agent import Agent
from sciql.utils.flax_utils import nonpytree_field
from sciql.algorithms.jax.offline_rl.sorl.samplers.samplers import MINE_Batch, SORL_Batch, Dummy_Sampler
from sciql.algorithms.jax.offline_rl.sorl.networks import GLCDiscreteActor, GLCActor, GLCDiscreteCritic, GLCValue
from sciql.algorithms.jax.offline_rl.sorl.encoders import gc_encoder_modules, GCEncoder, lc_encoder_modules

def update_by_loss_grad(train_state: TrainState, loss_fn: Callable) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss

def expectile_loss(diff, expectile=0.8) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class SORL_TrainState(NamedTuple):
    mine: TrainState
    task_critic: TrainState
    task_target_critic: TrainState
    task_value: TrainState
    actor: TrainState

class SORL_Config(BaseModel):
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
    layer_norm: bool = True
    gc_encoder: Optional[str] = None
    lc_encoder: Optional[str] = 'embedding'
    discrete: bool = False
    action_dim: Optional[int] = None

    # Algorithm
    label_prob_type: str = 'ind' # ind: indicator function ind(z=z_cur), mine: p(z)*exp(T(s,a,z)), prob_sigmoid: p(z|s,a) with sigmoid, prob_softmax: p(z|s,a) with softmax, 
    expectile: float = 0.7
    beta: float = 3.0
    tau: float =  0.005
    discount: float =  0.99
    normalize_state: bool = False
    train_task: bool = False

    # Optimization
    mine_lr: float = 3e-4
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    opt_decay_schedule: bool = True
    mine_n_jitted_updates: int = 1
    policy_max_steps: int = int(1e6)
    policy_n_jitted_updates: int = 1
    
class SORL_Agent(flax.struct.PyTreeNode, Agent):

    rng: jax.random.PRNGKey
    eval_rng: jax.random.PRNGKey
    eval_labels: Any = nonpytree_field()
    config: SORL_Config = nonpytree_field()
    train_state: SORL_TrainState
    obs_mean: Any = nonpytree_field()
    obs_std: Any = nonpytree_field()
    labels_numbers: Any = nonpytree_field()
    labels_probs: Any = nonpytree_field()
    batch_structure: Any = nonpytree_field()
    
    ##################
    # Initialization #
    ##################
    @classmethod
    def create(
        cls,
        sampler: Sampler,
        cfg: Optional[dict] = None,
        config: Optional[SORL_Config] = None,
    ):  
        # Get config from cfg or use the provided config object
        if config is None:
            if cfg is None: cfg = {}
            config = SORL_Config(**cfg)

        # Set jax seed
        rng = jax.random.PRNGKey(config.seed)
        rng, eval_rng, batch_rng, mine_rng, task_value_rng, task_critic_rng, actor_rng = jax.random.split(rng, 7)

        # Get data from batch
        example_batch: SORL_Batch = sampler.sample_batch(1, batch_rng)
        observations = example_batch.observations
        actions = example_batch.actions
        labels = example_batch.curr_labels
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
            gc_encoders['value'] = GCEncoder(state_encoder=gc_encoder_module())
            gc_encoders['critic'] = GCEncoder(state_encoder=gc_encoder_module())
            gc_encoders['actor'] = GCEncoder(state_encoder=gc_encoder_module())
        
        # Define lc_encoders
        lc_encoders = dict()
        if config.lc_encoder is not None:
            lc_encoder_module = lc_encoder_modules[config.lc_encoder]
            lc_encoders['mine'] = lc_encoder_module(labels_numbers=sampler.labels_numbers, label_embed_dim=config.label_embed_dim)
            lc_encoders['value'] = lc_encoder_module(labels_numbers=sampler.labels_numbers, label_embed_dim=config.label_embed_dim)
            lc_encoders['critic'] = lc_encoder_module(labels_numbers=sampler.labels_numbers, label_embed_dim=config.label_embed_dim)
            lc_encoders['actor'] = lc_encoder_module(labels_numbers=sampler.labels_numbers, label_embed_dim=config.label_embed_dim)
            
        # Initlialize style modules

        # MINE train state
        if config.label_prob_type != 'ind':

            if config.label_prob_type == 'mine':
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
            elif config.label_prob_type == 'prob_sigmoid':
                if config.discrete:
                    mine_model = GLCDiscreteCritic(
                        hidden_dims=config.hidden_dims,
                        gc_encoder=gc_encoders.get('mine'),
                        lc_encoder=lc_encoders.get('mine'),
                        action_dim=action_dim,
                        activate_final=True,
                        output_activation=nn.sigmoid
                    )
                else:
                    mine_model = GLCValue(
                        hidden_dims=config.hidden_dims,
                        gc_encoder=gc_encoders.get('mine'),
                        lc_encoder=lc_encoders.get('mine'),
                        activate_final=True,
                        output_activation=nn.sigmoid
                    )
            elif config.label_prob_type == 'prob_softmax':
                assert len(sampler.labels_numbers) == 1
                # To compute p(z|s,a), Take f(s,a) -> logits of dim (B, N_labels) -> softmax -> probs of dim (B, N_labels) -> p(z|s,a) = probs[z]
                if config.discrete:
                    mine_model = GLCDiscreteCritic(
                        hidden_dims=config.hidden_dims,
                        output_dim=sampler.labels_numbers[0],
                        squeeze_output=False,
                        gc_encoder=gc_encoders.get('mine'),
                        lc_encoder=lc_encoders.get('mine'),
                        action_dim=action_dim,
                        activate_final=True,
                        output_activation=nn.softmax
                    )
                else:
                    mine_model = GLCValue(
                        hidden_dims=config.hidden_dims,
                        output_dim=sampler.labels_numbers[0],
                        squeeze_output=False,
                        gc_encoder=gc_encoders.get('mine'),
                        lc_encoder=lc_encoders.get('mine'),
                        activate_final=True,
                        output_activation=nn.softmax
                    )
            
            mine = TrainState.create(
                apply_fn=mine_model.apply,
                params=mine_model.init(mine_rng, observations, actions=actions, labels=labels),
                tx=optax.adam(learning_rate=config.mine_lr),
            )
        else:
            mine = None
        
        if config.train_task:
            
            # Value train state
            value_model = GLCValue(
                hidden_dims=config.hidden_dims,
                layer_norm=config.layer_norm,
                gc_encoder=gc_encoders.get('value'),
                lc_encoder=lc_encoders.get('value'),
            )
            task_value = TrainState.create(
                apply_fn=value_model.apply,
                params=value_model.init(task_value_rng, observations),
                tx=optax.adam(learning_rate=config.value_lr),
            )

            # Critic train state
            if config.discrete:
                critic_model = GLCDiscreteCritic(
                    hidden_dims=config.hidden_dims,
                    ensemble=True,
                    gc_encoder=gc_encoders.get('critic'),
                    lc_encoder=lc_encoders.get('critic'),
                    action_dim=action_dim,
                )
            else:
                critic_model = GLCValue(
                    hidden_dims=config.hidden_dims,
                    ensemble=True,
                    gc_encoder=gc_encoders.get('critic'),
                    lc_encoder=lc_encoders.get('critic'),
                )
            task_critic = TrainState.create(
                apply_fn=critic_model.apply,
                params=critic_model.init(task_critic_rng, observations, actions=actions),
                tx=optax.adam(learning_rate=config.critic_lr),
            )
            task_target_critic = TrainState.create(
                apply_fn=critic_model.apply,
                params=critic_model.init(task_critic_rng, observations, actions=actions),
                tx=optax.adam(learning_rate=config.critic_lr),
            )
        else:
            task_value = task_critic = task_target_critic = None

        # Initilize actor module
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
        train_state = SORL_TrainState(
            mine=mine,
            task_critic=task_critic,
            task_target_critic=task_target_critic,
            task_value=task_value,
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
            labels_probs=sampler.labels_probs,
            batch_structure=batch_structure
        )

    ####################
    # Saving & Loading #
    ####################
    def save(self, path: str):
        """
        Saves the agent by converting the data structure to a dict for
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
            'labels_probs': self.labels_probs.tolist(),
            'config': self.config.model_dump()
        }

        # 5. Save attributes
        metadata_path = os.path.join(path, 'agent_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Agent and metadata successfully saved to: {path}")
        return self

    @classmethod
    def load(cls, path: str) -> "SORL_Agent":
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
        data_structure = SORL_Batch(**reconstructed_dict)

        # 4. Retrieve other sampler's attributes
        obs_mean = float(metadata['obs_mean'])
        obs_std = float(metadata['obs_std'])
        labels_numbers = tuple(metadata['labels_numbers'])
        labels_probs = jnp.array(metadata['labels_probs'])

        # 5. Create the config object from the loaded dictionary
        config = SORL_Config(**metadata['config'])
        
        # 6. Create the abstract agent.
        abstract_agent = jax.eval_shape(
            lambda: cls.create_abstract(
                data_structure=data_structure,
                obs_mean=obs_mean,
                obs_std=obs_std,
                labels_numbers=labels_numbers,
                labels_probs=labels_probs,
                config=config  # Pass the loaded config
            )
        )

        # 7. Restore the agent state.
        checkpointer = ocp.PyTreeCheckpointer()
        return checkpointer.restore(path, item=abstract_agent)

    @classmethod
    def create_abstract(
        cls, 
        data_structure: Any, 
        obs_mean: float, 
        obs_std: float, 
        labels_numbers: tuple,
        labels_probs: list, 
        config: SORL_Config
    ):
        """Internal helper that creates a dummy agent from data metadata."""
        dummy_sampler = Dummy_Sampler(
            data_structure=data_structure,
            obs_mean=obs_mean,
            obs_std=obs_std,
            labels_numbers=labels_numbers,
            labels_probs=labels_probs
        )
        return cls.create(sampler=dummy_sampler, config=config)
    
    #################
    # MINE Training #
    #################
    def update_mine_one_step(
        self,
        batch: MINE_Batch,
        train_state_mine: TrainState
    ) -> tuple[TrainState, dict]:
        """
        One mine step
        """
        def mine_loss_fn(params: flax.core.FrozenDict) -> jnp.ndarray:
            
            if self.config.label_prob_type == 'mine':
                t_joint = train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.joint_labels)
                t_marginal = train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.marginal_labels)

            elif self.config.label_prob_type == 'prob_sigmoid':

                sigma_joint = train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.joint_labels) + 1e-8
                p_joint = self.labels_probs[batch.joint_labels].squeeze(-1) + 1e-8
                t_joint = jnp.log(sigma_joint / p_joint)
                
                sigma_marginal = train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.marginal_labels) + 1e-8
                p_marginal = self.labels_probs[batch.marginal_labels].squeeze(-1) + 1e-8
                t_marginal = jnp.log(sigma_marginal / p_marginal)

            elif self.config.label_prob_type == 'prob_softmax':

                sigma_joint = jnp.take_along_axis(train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.joint_labels), batch.joint_labels, axis=1).squeeze(axis=1) + 1e-8
                p_joint = self.labels_probs[batch.joint_labels].squeeze(-1) + 1e-8
                t_joint = jnp.log(sigma_joint / p_joint)
                
                sigma_marginal = jnp.take_along_axis(train_state_mine.apply_fn(params, batch.observations, actions=batch.actions, labels=batch.marginal_labels), batch.marginal_labels, axis=1).squeeze(axis=1) + 1e-8
                p_marginal = self.labels_probs[batch.marginal_labels].squeeze(-1) + 1e-8
                t_marginal = jnp.log(sigma_marginal / p_marginal)

            assert (t_joint.shape == t_marginal.shape == (batch.observations.shape[0],))

            mi_estimate = jnp.mean(t_joint) - jnp.log(jnp.mean(jnp.exp(t_marginal)) + 1e-8)
            return -mi_estimate

        new_mine_train_state, loss = update_by_loss_grad(train_state_mine, mine_loss_fn)
        return new_mine_train_state, loss

    @jax.jit
    def update_mine_n_steps(self, sampler: Sampler) -> tuple["SORL_Agent", dict]:
        """
        Loop on the jitted updates
        """
        rng = self.rng
        new_mine_train_state = self.train_state.mine
        total_loss = 0.0

        for _ in range(self.config.mine_n_jitted_updates):
            rng, batch_rng = jax.random.split(rng)
            batch = sampler.sample_mine_batch(self.config.batch_size, batch_rng)
            new_mine_train_state, step_loss = self.update_mine_one_step(batch, new_mine_train_state)
            total_loss += step_loss

        mean_loss = total_loss / self.config.mine_n_jitted_updates
        infos = {f'mine_loss': mean_loss}

        new_train_state = self.train_state._replace(mine=new_mine_train_state)
        return self.replace(train_state=new_train_state, rng=rng), infos
    
    ###################
    # Policy Training #
    ###################
    def update_task_critic(
            self, 
            train_state: SORL_TrainState, 
            batch: SORL_Batch
        ) -> Tuple["SORL_TrainState", Dict]:

        next_v = train_state.task_value.apply_fn(train_state.task_value.params, batch.next_observations)
        target_q = batch.task_rewards + self.config.discount * batch.task_masks * next_v
        
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            q1, q2 = train_state.task_critic.apply_fn(
                critic_params, batch.observations, actions=batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(train_state.task_critic, critic_loss_fn)
        return train_state._replace(task_critic=new_critic), {'loss': critic_loss}

    def update_task_value(
            self, 
            train_state: SORL_TrainState, 
            batch: SORL_Batch
        ) -> Tuple["SORL_TrainState", Dict]:

        q1, q2 = train_state.task_target_critic.apply_fn(train_state.task_target_critic.params, batch.observations, actions=batch.actions)
        q = jax.lax.stop_gradient(jnp.minimum(q1, q2))

        def value_loss_fn(value_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            v = train_state.task_value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, self.config.expectile).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(train_state.task_value, value_loss_fn)
        return train_state._replace(task_value=new_value), {'loss': value_loss}

    def update_actor_one_step(
        self,
        train_state: SORL_TrainState, # Actor needs access to all critics/values
        batch: SORL_Batch
    ) -> Tuple[TrainState, Dict]:
        
        # Initialize infos 
        infos = {}

        # Compute probs p(z|s,a) of shape (B,N)
        B, N = batch.observations.shape[0], self.labels_numbers[0]
        all_labels = jnp.arange(N)

        if self.config.label_prob_type != 'ind':

            if self.config.label_prob_type == 'mine':
                def f(ts, obs, act, label_idx): return ts.apply_fn(ts.params, obs, actions=act, labels=jnp.full((B, 1), label_idx))
                t_actor = jax.vmap(f, in_axes=(None, None, None, 0), out_axes=1)(train_state.mine, batch.observations, batch.actions, all_labels)
                p_actor = jnp.tile(self.labels_probs[all_labels], (B, 1)) * jnp.exp(t_actor)

            elif self.config.label_prob_type == 'prob_sigmoid':
                def f(ts, obs, act, label_idx): return ts.apply_fn(ts.params, obs, actions=act, labels=jnp.full((B, 1), label_idx))
                p_actor = jax.vmap(f, in_axes=(None, None, None, 0), out_axes=1)(train_state.mine, batch.observations, batch.actions, all_labels)
                
            elif self.config.label_prob_type == 'prob_softmax':
                def f(ts, obs, act, label_idx): return jnp.take_along_axis(ts.apply_fn(ts.params, obs, actions=act, labels=jnp.full((B, 1), label_idx)), jnp.full((B, 1), label_idx), axis=1).squeeze(axis=1) + 1e-8
                p_actor = jax.vmap(f, in_axes=(None, None, None, 0), out_axes=1)(train_state.mine, batch.observations, batch.actions, all_labels)
        else:
            p_actor = jax.nn.one_hot(batch.curr_labels, num_classes=N)

        # Compute rewards advantages
        if self.config.train_task:
            task_v = train_state.task_value.apply_fn(train_state.task_value.params, batch.observations)
            task_q = jnp.minimum(*train_state.task_target_critic.apply_fn(train_state.task_target_critic.params, batch.observations, actions=batch.actions))
            task_advs = jnp.stack([task_q - task_v], axis=-1)
        else:
            task_advs = jnp.zeros((self.config.batch_size, 1))
        infos.update({
            'task_adv_mean':jnp.mean(task_advs),
            'task_adv_min':jnp.min(task_advs),
            'task_adv_max':jnp.max(task_advs)
        })
        
        # Ensure the shape of the advantage and compute weight
        task_advs = task_advs.reshape(B, 1)
        beta_adv = task_advs * self.config.beta
        exp_a = jnp.exp(beta_adv)
        clipped_exp_a = jnp.minimum(exp_a, 100.0)
        clipped_exp_a = jnp.tile(clipped_exp_a, (1, N))
        weights = p_actor * clipped_exp_a

        def actor_loss_fn(actor_params: flax.core.FrozenDict) -> jnp.ndarray:

            def logprob_for_label(label_idx):
                dist = train_state.actor.apply_fn(actor_params, batch.observations, labels=jnp.full((B, 1), label_idx))
                return dist.log_prob(batch.actions).reshape(-1)

            log_probs_BN = jax.vmap(logprob_for_label, in_axes=0, out_axes=1)(all_labels)  # (B, N)
            actor_loss = -(weights * log_probs_BN).mean()
            return actor_loss

        new_actor, a_loss = update_by_loss_grad(train_state.actor, actor_loss_fn)
        infos.update({
            'loss': a_loss,
            'beta_adv_mean': jnp.mean(beta_adv),
            'beta_adv_min': jnp.min(beta_adv),
            'beta_adv_max': jnp.max(beta_adv),
            'exp_a_mean': jnp.mean(exp_a),
            'exp_a_min': jnp.min(exp_a),
            'exp_a_max': jnp.max(exp_a),
            'clipped_exp_a_mean': jnp.mean(clipped_exp_a),
            'clipped_exp_a_min': jnp.min(clipped_exp_a),
            'clipped_exp_a_max': jnp.max(clipped_exp_a)
        })
        return new_actor, infos
    
    def target_update(self, model: TrainState, target_model: TrainState, tau: float) -> TrainState:
        new_target_params = jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params)
        return target_model.replace(params=new_target_params)

    @jax.jit
    def update_policy_one_step_jitted(
        self,
        train_state: "SORL_TrainState",
        batch: SORL_Batch,
        rng: jax.random.PRNGKey
    ) -> Tuple["SORL_TrainState", Dict, jax.random.PRNGKey]:
        """
        Performs exactly ONE update step for ALL models (values, critics, actor).
        This function is JIT-compiled and contains the internal loops over the different models.
        """
        
        all_losses = {}

        # Update all task value, critic and target_critic networks
        if self.config.train_task:
            train_state, value_infos = self.update_task_value(train_state, batch)
            all_losses.update({f'value_task_{k}': v for k, v in value_infos.items()})

            train_state, critic_infos = self.update_task_critic(train_state, batch)
            all_losses.update({f'critic_task_{k}': v for k, v in critic_infos.items()})

            new_target_critic = self.target_update(train_state.task_critic, train_state.task_target_critic, self.config.tau)
            train_state = train_state._replace(task_target_critic=new_target_critic)

        # Now, update the actor using this intermediate state
        new_actor, actor_infos = self.update_actor_one_step(train_state, batch)
        all_losses.update({f'actor_{k}': v for k, v in actor_infos.items()})
        
        # Assemble the final training state
        train_state = train_state._replace(actor=new_actor)
        
        return train_state, all_losses

    @jax.jit
    def update_policy_n_steps(self, sampler: Sampler) -> Tuple["SORL_Agent", Dict]:
        """
        Main training function for the policy. Contains the outer Python loop over training steps.
        """
        # Set rng and train_state
        current_rng = self.rng
        current_train_state = self.train_state
        
        # Create an accumulator for losses as a pytree
        loss_accumulator_template = {
            # Value
            'value_task_loss': 0.0,
            # Critic
            'critic_task_loss': 0.0,
            # Actor raw advantages
            'actor_task_adv_mean': 0.0,
            'actor_task_adv_min': 0.0,
            'actor_task_adv_max': 0.0,
            # Actor gated advantages
            'actor_loss': 0.0,
            'actor_beta_adv_mean': 0.0,
            'actor_beta_adv_min': 0.0,
            'actor_beta_adv_max': 0.0,
            'actor_exp_a_mean': 0.0,
            'actor_exp_a_min': 0.0,
            'actor_exp_a_max': 0.0,
            'actor_clipped_exp_a_mean': 0.0,
            'actor_clipped_exp_a_min': 0.0,
            'actor_clipped_exp_a_max': 0.0,
        }
        total_losses = jax.tree_util.tree_map(lambda x: x, loss_accumulator_template)

        # Train over all jitted steps
        for _ in range(self.config.policy_n_jitted_updates):
            current_rng, batch_rng, update_rng = jax.random.split(current_rng, 3)
            batch = sampler.sample_batch(self.config.batch_size, batch_rng)
            current_train_state, step_losses = self.update_policy_one_step_jitted(current_train_state, batch, update_rng)
            total_losses = jax.tree_util.tree_map(lambda x, y: x + y, total_losses, step_losses)
    
        # Calculate the mean loss over all the steps
        mean_losses = jax.tree_util.tree_map(
            lambda x: x / self.config.policy_n_jitted_updates, total_losses
        )
        
        return self.replace(train_state=current_train_state, rng=current_rng), mean_losses
    
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

    def to(self, device: str) -> "SORL_Agent":
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


