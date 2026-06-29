import cv2
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Any, List
from gymnasium.wrappers import TimeLimit
from sciql.core.data import Frame
from sciql.core.agent import Agent
from sciql.core.data import BaseEpisode
from sciql.core.env import Action, Observation, GymEnv
from sciql.envs.diverse_mujoco.statistics import Return, DiscountedReturn, NormalizedReturn, Length
from sciql.envs.diverse_mujoco.labels.base import get_episode_labels
from sciql.envs.diverse_mujoco.base_envs.core import envs, time_limits

class DiverseMujoco_Environment(GymEnv):

    def __init__(
        self,
        env_name: str,
        env_kwargs: dict = {},
    ):
        super().__init__()
        
        self.env_name = env_name
        self.max_steps = time_limits[env_name]
        assert env_name in envs, f'Supported environments are {envs.keys()}'
        self.gym_env = TimeLimit(envs[env_name](**env_kwargs), max_episode_steps=self.max_steps)
        self.statistics = {
            'return': Return(),
            'discounted_return': DiscountedReturn(discount=0.99),
            'length': Length(),
            'normalized_return': NormalizedReturn(env_name=env_name)
        }
        self.labels = {label.name: label for label in get_episode_labels(env_name)}
    
    def reset(self, seed: int = None):
        np_observation, infos = self.gym_env.reset(seed=seed)
        return {'full': np_observation}, infos

    def sample_action(self):
        return {'full': self.gym_env.action_space.sample()}
    
    def step(self, action: Action):
        next_observation, reward, terminated, truncated, infos = self.gym_env.step(action['full'])
        return (
            {'full': next_observation},
            np.array(reward),
            np.array(terminated),
            np.array(truncated),
            infos
        )
    
    def render(self, headless: bool = False):
        img = self.gym_env.render()
        if not headless:
            cv2.imshow(f"{self.env_name}", img[:,:,::-1])
            cv2.waitKey(16) # 16ms ~ 62.5 fps
        return img

    def generate_episodes(
        self,
        agent: Agent,
        task: str = None,
        seed: int = None,
        device: str = None,
        n_episodes: int = 1,
        n_video_episodes: int = 0,
        agent_reset_args: Union[DictConfig, Dict[str, Any]] = None,
        agent_eval_mode_args: Union[DictConfig, Dict[str, Any]] = None,
        verbose: bool = False,
        render: bool = False
    ) -> List[BaseEpisode]:
        
        # Initialize gym_env and agent
        observation, infos = self.reset(seed=seed)
        agent = agent.reset(seed, **agent_reset_args)
        agent = agent.set_eval_mode(**agent_eval_mode_args)
        if device is not None: agent = agent.to(device)
        episodes, videos = [], []

        for i in tqdm(range(n_video_episodes + n_episodes), desc='Generating episodes', disable=not verbose):

            # Reset
            episode = BaseEpisode()
            video_episode = (i < n_video_episodes)
            if video_episode: video = []
            episode_step = 0
            episode_pbar = tqdm(total=1000, desc=f"Generating episode {i}")
            
            # Generate
            while True:
                
                # Render
                if render or video_episode:
                    img = self.render(headless=not render)
                    if video_episode: video.append(img)

                # Step
                agent, action = agent.act(observation)
                next_observation, reward, terminated, truncated, next_infos = self.step(action)
                
                # Add frame to episode
                episode.add_frame(Frame(observation, action, reward, next_observation, terminated, truncated, infos))

                episode_step += 1
                episode_pbar.update(1)

                if terminated or truncated:
                    seed += 1
                    agent.reset(seed=seed, **agent_reset_args)
                    observation, infos = self.reset(seed=seed)
                    break
                else:
                    observation = next_observation
                    infos = next_infos
            
            episode_pbar.close()
            
            # Store
            episodes.append(episode)
            if video_episode: videos.append(video)
            cv2.destroyAllWindows() # This is good practice if you were displaying windows

        return episodes, videos