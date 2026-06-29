import numpy as np
from typing import Dict, Union, Optional, Any, List
from sciql.core.agent import Agent
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from sciql.core.env import Action, Observation, GymEnv, Env

class RandomAgent(Agent):

    def __init__(
        self,
        env: Env,
    ):
        super().__init__()
        self.env = env
    
    def reset(self, *args, **kwargs):
        return self

    def set_eval_mode(self, **kwargs):
        return self
    
    def set_train_mode(self, **kwargs):
        return self

    def to(self):
        return self

    def act(self, observation, **kwargs):
        return self, self.env.sample_action()
    
class CircleFollowingAgent(Agent):
    def __init__(
        self,
        target_radius_env: float, 
        target_direction: int, 
        env_step_size: float,
        env_action_scale: float,
        action_noise_std_dev_rad: float = 0.0,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.target_radius_env = target_radius_env
        self.target_direction = target_direction # 1 for CCW, -1 for CW
        self.env_step_size = env_step_size
        self.env_action_scale = env_action_scale # From Traj2D environment
        self.action_noise_std_dev_rad = action_noise_std_dev_rad
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, **kwargs): # kwargs for agent_reset_args
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Re-initialize if target parameters change per reset, 
        # but for this problem, they are set per episode.
        # So, this reset mainly handles the RNG.

    def set_eval_mode(self, **kwargs): # kwargs for agent_eval_mode_args
        # For this simple agent, eval mode might mean no noise, but we control noise at init.
        pass

    def set_train_mode(self, **kwargs):
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        # Observation is not strictly needed for this fixed policy agent,
        # but good practice to have it in the signature.

        if self.target_radius_env < 1e-4: # Avoid division by zero
            ideal_delta_theta_agent_rad = 0.0
        else:
            ideal_delta_theta_agent_rad = self.target_direction * (self.env_step_size / self.target_radius_env)
        
        noise_rad = self.rng.normal(0, self.action_noise_std_dev_rad)
        noisy_delta_theta_agent_rad = ideal_delta_theta_agent_rad + noise_rad
        
        # Convert to environment action space [-1, 1]
        env_action_val = noisy_delta_theta_agent_rad / self.env_action_scale
        env_action = np.clip(np.array([env_action_val]), -1.0, 1.0)
        return {'full': env_action}

    def to(self, device: str):
        # This agent is numpy-based, so this is a no-op unless you adapt it for PyTorch models
        return self