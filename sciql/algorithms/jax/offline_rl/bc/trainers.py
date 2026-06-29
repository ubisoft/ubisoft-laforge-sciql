import os
from tqdm import tqdm
from omegaconf import DictConfig
from sciql.core.trainer import SingleLearningTrainer
from sciql.core.agent import Agent, AgentsDB
from sciql.core.data import Sampler
from sciql.core.logger import Logger
from sciql.utils.seeding.seeding import set_seed
from sciql.utils.seeding.jax_seeding import set_jax_determinism
from sciql.utils.imports import get_class, instantiate_class

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

class BC_Trainer(SingleLearningTrainer):

    def train(
        algo_cfg: DictConfig,
        agent_cfg: DictConfig,
        agents_db: AgentsDB,
        logger: Logger
    ):
        
        # Seeding
        set_seed(algo_cfg.seed)
        set_jax_determinism(algo_cfg.jax_deterministic)

        # Prepare sampler (jittable or not jittable, but you can sample from it)
        sampler: Sampler = get_class(algo_cfg.sampler).create(algo_cfg.sampler)

        # Prepare agent
        agent: Agent = get_class(agent_cfg.agent).create(sampler, agent_cfg.agent)

        # Restore agent if needed
        if ('restore' in algo_cfg and algo_cfg.restore is not None):
            restore_agents_db = instantiate_class(algo_cfg.restore.agents_db)
            agent = restore_agents_db.get_last_stage(algo_cfg.restore.agent_id, agent)
            
        # Training
        if agent_cfg.agent.policy_jitted_sampling:
            n_updates = agent_cfg.agent.policy_n_jitted_updates
            num_steps = agent_cfg.agent.policy_max_steps // n_updates
        else:
            n_updates = 1
            num_steps = agent_cfg.agent.policy_max_steps
        
        with tqdm(total=agent_cfg.agent.policy_max_steps, smoothing=0.1, dynamic_ncols=True, disable=not algo_cfg.verbose, desc='Training') as progress_bar_train:

            for i in range(1, num_steps + 1):
                
                # Update agent
                gradient_step = i * n_updates
                if agent_cfg.agent.policy_jitted_sampling:
                    agent, infos = agent.update_n_steps(sampler)
                else:
                    agent, infos = agent.update(sampler)
                    
                # Log metrics
                if (gradient_step == n_updates) or (gradient_step % algo_cfg.log_interval == 0):
                    train_metrics = {f'training/{k}': v.item() for k, v in infos.items()}
                    for k,v in train_metrics.items(): logger.add_scalar(k, v, gradient_step)

                # Evaluate agent
                if (gradient_step == n_updates and algo_cfg.eval_init) or gradient_step % algo_cfg.eval_interval == 0:
                    yield agent, {'gradient_step': gradient_step}

                # Save agent
                if (gradient_step == n_updates) or (gradient_step % algo_cfg.save_interval == 0):
                    agents_db.add_agent(agent, 'agent', gradient_step)

                # Update bar
                progress_bar_train.update(agent_cfg.agent.policy_n_jitted_updates)



            


        


        
