import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sciql.core.data import Frame
from sciql.envs.traj2d.traj2d import Traj2D
from typing import Dict, Union, Optional, Any, List
from omegaconf import DictConfig
from sciql.core.agent import Agent
from sciql.core.data import BaseEpisode
from tqdm import tqdm
from sciql.core.env import Action, Observation, GymEnv, Env
from sciql.envs.traj2d.statistics.base import get_traj2d_episode_statistics
from sciql.envs.traj2d.labels.base import get_traj2d_episode_labels

class Traj2D_Environment(GymEnv):

    def __init__(
        self,
        num_modes: int = 4,
        scale: float = 1.,
        has_limits: bool = True,
        limits: list = [[-50.0, 50.0], [-50.0, 50.0]],
        terminates_on_limits: bool = False,
        history: int = 4,
        include_start_state: bool = True,
        min_speed: float = 0.5,
        max_speed: float = 3.0,
        visualize_reward: bool = True,
        reward_grid_res: int = 220,
        reward_cmap: str = "viridis",
        reward_alpha: float = 0.55,
        reward_contrast: float = 20.0,  # real units span for colormap normalization

    ):
        super().__init__()
        self.gym_env = Traj2D(
            num_modes=num_modes, 
            scale=scale,
            has_limits=has_limits,
            limits=limits,
            terminates_on_limits=terminates_on_limits,
            history=history,
            include_start_state=include_start_state,
            min_speed=min_speed,
            max_speed=max_speed,
            visualize_reward=visualize_reward,
            reward_grid_res=reward_grid_res,
            reward_cmap=reward_cmap,
            reward_alpha=reward_alpha,
            reward_contrast=reward_contrast,  # real units span for colormap normalization
        )
        self.statistics = {statistic.name: statistic for statistic in get_traj2d_episode_statistics()}
        self.labels = {label.name: label for label in get_traj2d_episode_labels()}
    
    def reset(
        self,
        mode_idx: int = 0,
        random_start_x: bool = True,
        start_x_real: float = 0.0,
        random_start_y: bool = True,
        start_y_real: float = 0.0,
        random_initial_theta: bool = False,
        initial_theta_real: float = 0.0,
        seed: Optional[int] = None,
        **kwargs
    ):  
        reset_args = {
            "mode_idx": mode_idx,
            "random_start_x": random_start_x,
            "start_x_real": start_x_real,
            "random_start_y": random_start_y,
            "start_y_real": start_y_real,
            "random_initial_theta": random_initial_theta,
            "initial_theta_real": initial_theta_real,
        }
        obs, info = self.gym_env.reset(seed=seed, options=reset_args)
        return {'full': obs}, info
    
    def sample_action(self):
        return {'full': self.gym_env.action_space.sample()}
    
    def step(self, action: Action) -> Frame:
        next_observation, reward, terminated, truncated, infos = self.gym_env.step(action['full'])
        return (
            {'full': next_observation},
            np.array(reward),
            np.array(terminated),
            np.array(truncated),
            infos
        )

    def render(self, headless=False, camera_mode='fixed'):
        """
        If headless=False, renders a non-blocking, moving plot (interactive).
        If headless=True, returns a (H, W, 3) RGB image without displaying it.
        """
        return self.gym_env.render(headless, camera_mode)
        
    def generate_episodes(
        self,
        agent: Agent,
        task: str = None,
        seed: int = 0,
        device: str = None,
        n_episodes: int = 1,
        n_video_episodes: int = 0,
        env_reset_args: Union[DictConfig, Dict[str, Any]] = {},
        agent_reset_args: Union[DictConfig, Dict[str, Any]] = {},
        agent_eval_mode_args: Union[DictConfig, Dict[str, Any]] = {},
        verbose: bool = False,
        render: bool = False
    ) -> List[BaseEpisode]:
        
        # Initialize gym_env and agent
        observation, infos = self.reset(seed=seed, **env_reset_args)
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
            episode_pbar = tqdm(total=1000, desc=f"Generating episode {i}", disable=not verbose)

            # Generate
            while True:
                
                # Render
                if render or video_episode:
                    img = self.render(headless=not render)
                    if video_episode: video.append(img)

                agent, action = agent.act(observation)
                next_observation, reward, terminated, truncated, next_infos = self.step(action)

                # Add frame to episode
                episode.add_frame(Frame(observation, action, reward, next_observation, terminated, truncated, infos))

                episode_step += 1
                episode_pbar.update(1)

                if terminated or truncated:
                    seed += 1
                    agent.reset(seed=seed, **agent_reset_args)
                    observation, infos = self.reset(seed=seed, **env_reset_args)
                    break
                else:
                    observation = next_observation
                    infos = next_infos

            # Store
            episodes.append(episode)
            if video_episode: videos.append(video)
            cv2.destroyAllWindows() # This is good practice if you were displaying windows
        
        return episodes, videos