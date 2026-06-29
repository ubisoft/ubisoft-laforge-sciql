import numpy as np
import random as rd

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Dict, Union, Optional, Any, List, Dict, Tuple
from sciql.core.agent import Agent
from sciql.core.data import Observation, Action, Frame, Episode
from sciql.core.statistic import EpisodeStatistic

class Env(ABC):
    """
    An `Env` is a class that allows sequential interactions with 
    an `Agent` to generate episodes through the generate_episodes
    function. It comes with episodes statistics to analyse the 
    generated episodes.
    """
    def __init__(self) -> None:
        # statistic specific to the environment
        self.statistics: Dict[EpisodeStatistic] = {}
        
    @abstractmethod
    def generate_episodes(
        self,
        agent: Agent,
        task: str = None,
        seed: int = None,
        n_episodes: int = 1,
        max_episode_steps: int = None,
        agent_reset_args: Union[DictConfig, Dict[str, Any]] = None,
        agent_eval_mode_args: Union[DictConfig, Dict[str, Any]] = None,
        verbose: bool = False
    ) -> List[Episode]:
        """
        Generates a number of episodes from an interaction between an agent and the environment.

        Args:
            agent (Agent): The agent to interact with.
            task (str): The task name to specify the mode of the environment.
            seed (int): The seed used for the generation, without reset of the seed at each episode because of seed incrementation.
            n_episodes (int): The number of episodes to generate.
            max_episode_steps (int): The max number of episode steps.
            agent_reset_args (Union[DictConfig, Dict[str, Any]): The arguments for the reset of the agent.
            agent_eval_mode_args (Union[DictConfig, Dict[str, Any]): The arguments for the eval mode of the agent.
            verbose (bool): If the generation prints updates.

        Returns:
            (List[Episode]): The list of the generated episodes.
        """
        raise NotImplementedError

class GymEnv(Env):
    """
    A `GymEnv` is a Env that can be written in a gymnasium like format. This
    is the case for gym environments, but not for more complex environments
    like Godot environments.
    """
    @abstractmethod
    def reset(self, seed: int = None, **kwargs):
        """
        Resets the environment according to the current seeding.

        Args:
            seed (int): The seed used for the generation.

        Returns:
            observation (Observation): The new current observation. e.g. observation['full'] is a np.ndarray of shape (obs_dim,)
            infos (Dict): The possible reset informations.
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_action(self) -> np.ndarray:
        """
        Returns a representative action of the environment action space.

        Returns:
            (Action): A random action. e.g. action['full'] is a np.ndarray of shape (act_dim,)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Union[Action, Dict[str, np.ndarray]]) -> Tuple[Frame, Dict]:
        """
        Proceed to one environment step, using action 
        information provided by an action.
        Then it returns the current state of the `Env`.

        Args:
            action (Action): The action used to modify the environment state.

        Returns:
            next_observation (Observation): The next state observation
            reward (np.ndarray): The reward of the transition.
            terminated (np.ndarray): If the next_observation is a terminal state.
            truncated (np.ndarray): If the episode number of steps is >= max steps.
            info (Dict): The possible step informations.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, headless: bool = False) -> np.ndarray:
        """
        Procuces a rendering of the environment as a numpy array.

        Args:
            headless (bool): Wether the print the rendering.
            
        Returns:
            (np.ndarray): Rendering of the environment.
        """
        raise NotImplementedError
