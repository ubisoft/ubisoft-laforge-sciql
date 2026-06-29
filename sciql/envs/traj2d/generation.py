import argparse
import numpy as np
from tqdm import tqdm
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle

# If you keep BaseEpisode/Frame API, we reuse it
from sciql.core.data import BaseEpisode, Frame
from sciql.core.agent import Agent
from sciql.envs.traj2d.envs import Traj2D_Environment


# ---------------------------------------------------------------------
# Agent producing normalized actions for the new Traj2D env
#   - obs['full'] starts with [x_t, y_t, θ_t, ...]
#   - action['full'] = [a_turn, a_step] in [-1,1]^2
#       Δθ = a_turn * max_delta_theta
#       step_dist_env = (min_speed + (a_step+1)/2 * (max_speed-min_speed)) / scale
# ---------------------------------------------------------------------
class StyledTrajectoryAgent(Agent):
    """
    Two-mode controller:
      1) GOTO_CIRCLE: head to the closest point on the circle.
      2) FOLLOW_CIRCLE: tangent motion + P radial correction.

    Expects obs as dict or array; when dict, uses obs['full'].
    Returns action dict: {'full': np.array([a_turn, a_step], dtype=np.float32)} with values in [-1, 1].
    """
    def __init__(self,
                 target_center_real: np.ndarray,
                 target_radius_real: float,
                 turn_direction: int,          # +1 CCW, -1 CW
                 target_speed_real: float,     # desired real speed
                 curvature_noise_std_dev: float,
                 env_scale: float,
                 env_min_speed: float,
                 env_max_speed: float,
                 env_max_delta_theta: float,   # radians
                 seed: int):
        self.c = np.array(target_center_real, dtype=float)
        self.R = float(target_radius_real)
        self.turn = int(turn_direction)
        self.v_des = float(target_speed_real)
        self.noise_std = float(curvature_noise_std_dev)

        self.scale = float(env_scale)
        self.vmin = float(env_min_speed)
        self.vmax = float(env_max_speed)
        self.max_dtheta = float(env_max_delta_theta)

        self.rng = np.random.default_rng(seed)

        self.mode = 'GOTO_CIRCLE'
        self.enter_follow_tol = 5.0   # [real units]
        self.exit_follow_tol  = 10.0  # [real units]
        self.kp = 0.5                 # radial correction gain

    @staticmethod
    def _unwrap_obs(obs):
        if isinstance(obs, dict):
            return obs['full']
        return obs

    @staticmethod
    def _wrap(angle: float) -> float:
        return (angle + np.pi) % (2*np.pi) - np.pi

    def act(self, observation) -> dict:
        full = self._unwrap_obs(observation)
        # Newest state first: [x_t, y_t, θ_t, ...]
        x_t, y_t, th_t = map(float, full[:3])
        p_env = np.array([x_t, y_t], dtype=float)
        p_real = p_env * self.scale

        vec_c = self.c - p_real
        d = float(np.linalg.norm(vec_c))
        err = abs(d - self.R)

        # Mode switching (hysteresis)
        if self.mode == 'GOTO_CIRCLE' and err < self.enter_follow_tol:
            self.mode = 'FOLLOW_CIRCLE'
        elif self.mode == 'FOLLOW_CIRCLE' and err > self.exit_follow_tol:
            self.mode = 'GOTO_CIRCLE'

        if self.mode == 'GOTO_CIRCLE':
            if d > 1e-9:
                tgt = self.c - (vec_c / d) * self.R
            else:
                tgt = self.c + np.array([self.R, 0.0])
            theta_des = float(np.arctan2(tgt[1] - p_real[1], tgt[0] - p_real[0]))
        else:
            ang = float(np.arctan2(p_real[1] - self.c[1], p_real[0] - self.c[0]))
            n_rx, n_ry = np.cos(ang), np.sin(ang)
            if self.turn == 1:   # CCW tangent
                t_tx, t_ty = -np.sin(ang),  np.cos(ang)
            else:                # CW tangent
                t_tx, t_ty =  np.sin(ang), -np.cos(ang)
            radial_err = self.R - d
            vx = t_tx + self.kp * radial_err * n_rx
            vy = t_ty + self.kp * radial_err * n_ry
            theta_des = float(np.arctan2(vy, vx))

        # Desired relative turn Δθ
        delta_theta = self._wrap(theta_des - th_t)
        # Curvature noise applies to the turn command (Δθ)
        delta_theta += float(self.rng.normal(0.0, self.noise_std))
        delta_theta = self._wrap(delta_theta)

        # Map to normalized actions
        a_turn = np.clip(delta_theta / max(self.max_dtheta, 1e-8), -1.0, 1.0)

        v_cmd = float(np.clip(self.v_des, self.vmin, self.vmax))
        a_step = (v_cmd - self.vmin) / (self.vmax - self.vmin) * 2.0 - 1.0
        a_step = float(np.clip(a_step, -1.0, 1.0))

        return self, {'full': np.array([a_turn, a_step], dtype=np.float32)}

    def reset(self, seed: int = None, **kwargs):
        return self
    
    def set_eval_mode(self, **kwargs) -> "Agent":
        """
        Sets the policy to evaluation mode.

        Returns:
            (Agent): The agent with its policy set to evaluation mode.
        """
        return self

# ---------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------
def generate_styled_episodes(
    env,  # Traj2D_Environment instance (or the underlying env with same API)
    n_episodes: int = 10,
    steps_per_episode: int = 1000,
    seed: int = 0,
    # Generation parameter ranges (real units)
    start_x_range = (-30.0, 30.0),
    start_y_range = (-30.0, 30.0),
    center_x_range = (-30.0, 30.0),
    center_y_range = (-30.0, 30.0),
    radius_range = (3.0, 10.0),
    speed_range = (0.5, 3.0),
    curvature_noise_range = (0.0, 0.05),
    only_circles: bool = True,
    verbose: bool = False,
    render: bool = False,
) -> List[BaseEpisode]:

    episodes: List[BaseEpisode] = []
    rng = np.random.default_rng(seed)

    # Pull env attributes safely
    base = getattr(env, 'gym_env', env)
    env_scale = getattr(base, 'scale', 1.0)
    env_min_speed = getattr(base, 'min_speed', 0.5)
    env_max_speed = getattr(base, 'max_speed', 3.0)
    env_max_delta_theta = getattr(base, 'max_delta_theta', np.pi)

    for i in tqdm(range(n_episodes), desc='Generating styled episodes', disable=not verbose):
        cur_seed = seed + i
        ep = BaseEpisode()

        # sample episode parameters (real units)
        start_theta_real = float(rng.uniform(-np.pi, np.pi))
        target_center_real = np.array([rng.uniform(*center_x_range), rng.uniform(*center_y_range)], dtype=float)
        target_radius_real = float(rng.uniform(*radius_range))
        turn_direction = int(rng.choice([1, -1]))
        target_speed_real = float(rng.uniform(*speed_range))
        curvature_noise_std_dev = float(rng.uniform(*curvature_noise_range))

        if only_circles:
            center_direction = rng.normal(size=2)
            center_direction /= np.linalg.norm(center_direction)
            start_pos_real = target_center_real + center_direction*target_radius_real
        else:
            start_pos_real = np.array([rng.uniform(*start_x_range), rng.uniform(*start_y_range)], dtype=float)

        if verbose:
            print(
                f"Episode {i}:"
                f"\n  start_pos_real = {start_pos_real}"
                f"\n  start_theta_real = {start_theta_real:.3f}"
                f"\n  target_center_real = {target_center_real}"
                f"\n  target_radius_real = {target_radius_real:.3f}"
                f"\n  turn = {'CCW' if turn_direction==1 else 'CW'}"
                f"\n  target_speed_real = {target_speed_real:.3f}"
                f"\n  curvature_noise_std_dev = {curvature_noise_std_dev:.4f}\n"
            )
        
        agent = StyledTrajectoryAgent(
            target_center_real=target_center_real,
            target_radius_real=target_radius_real,
            turn_direction=turn_direction,
            target_speed_real=target_speed_real,
            curvature_noise_std_dev=curvature_noise_std_dev,
            env_scale=env_scale,
            env_min_speed=env_min_speed,
            env_max_speed=env_max_speed,
            env_max_delta_theta=env_max_delta_theta,
            seed=cur_seed,
        )

        # reset env (Gymnasium API: obs, info)
        obs, _ = env.reset(
            seed=cur_seed,
            mode_idx=0,
            random_start_x=False,
            start_x_real=float(start_pos_real[0]),
            random_start_y=False, 
            start_y_real=float(start_pos_real[1]),
            random_initial_theta=False,
            initial_theta_real=start_theta_real
        )

        # ensure obs is a dict with 'full' for consistent downstream usage
        if not isinstance(obs, dict):
            obs = {'full': obs}

        for _ in range(steps_per_episode):
            _, action = agent.act(obs)  # dict with 'full' key (normalized Δθ, speed)
            next_obs, reward, terminated, truncated, infos = env.step(action)

            # ensure next_obs is dict for consistency
            if not isinstance(next_obs, dict):
                next_obs = {'full': next_obs}

            ep.add_frame(Frame(obs, action, reward, next_obs, terminated, truncated, infos))

            obs = next_obs
            if render:
                env.render(headless=False)
            if terminated or truncated:
                break

        episodes.append(ep)

    return episodes


# ---------------------------------------------------------------------
# Plot utility (reads x,y,θ from obs['full'])
# ---------------------------------------------------------------------
def plot_last_episode(episodes: List[BaseEpisode], env_scale: float,
                      center: np.ndarray = None, radius: float = 0.0,
                      title: str = "Last episode"):
    if not episodes:
        return
    ep = episodes[-1]
    xs, ys = [], []
    for fr in ep.frames:
        ob_full = fr.observation['full'] if isinstance(fr.observation, dict) else fr.observation
        xy_t = ob_full[:2]
        pos_real = xy_t * env_scale
        xs.append(float(pos_real[0])); ys.append(float(pos_real[1]))

    fig, ax = plt.subplots()
    ax.plot(xs, ys, linewidth=1.5, label="trajectory")
    if center is not None and radius > 0:
        ax.add_patch(MplCircle(center, radius, fill=False, linestyle='--', label='target'))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title); ax.set_xlabel("x (real)"); ax.set_ylabel("y (real)")
    ax.legend(); plt.show()


def generate_and_save(env, args):

    try:
        episodes = generate_styled_episodes(
            env=env,
            n_episodes=args.n_episodes,
            steps_per_episode=args.steps_per_episode,
            seed=args.seed,
            start_x_range=(args.start_x_min_real, args.start_x_max_real),
            start_y_range=(args.start_y_min_real, args.start_y_max_real),
            center_x_range=(args.center_x_min_real, args.center_x_max_real),
            center_y_range=(args.center_y_min_real, args.center_y_max_real),
            radius_range=(args.min_radius_real, args.max_radius_real),
            speed_range=(args.min_speed, args.max_speed),
            curvature_noise_range=(args.min_curvature, args.max_curvature),
            only_circles=args.only_circles,
            verbose=args.verbose,
            render=args.render,
        )
    finally:
        if hasattr(env, 'close'):
            env.close()

    print(f"\nGenerated {len(episodes)} episode(s).")

    if args.plot and episodes:
        scale = getattr(getattr(env, 'gym_env', env), 'scale', 1.0)
        plot_last_episode(episodes, env_scale=scale, center=None, radius=0.0,
                          title="Last episode (trajectory only)")

    # Save to EpisodesDB
    episodes_db = Traj2D_EpisodesDB(".")
    for i, episode in enumerate(episodes):
        episodes_db.add_episode(episode)
    print("Saved episodes to EpisodesDB")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == '__main__':

    from sciql.data.episodes_db.traj2d.episodes_db import Traj2D_EpisodesDB

    parser = argparse.ArgumentParser(description="Generate circular trajectories (Δθ + normalized step).")
    parser.add_argument('--n_episodes', type=int, default=1000, help="Number of episodes.")
    parser.add_argument('--steps_per_episode', type=int, default=1000, help="Max steps per episode.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env_scale', type=float, default=1.0)

    parser.add_argument('--start_x_min_real', type=float, default=-30.0)
    parser.add_argument('--start_x_max_real', type=float, default=30.0)
    parser.add_argument('--start_y_min_real', type=float, default=-30.0)
    parser.add_argument('--start_y_max_real', type=float, default=30.0)

    parser.add_argument('--center_x_min_real', type=float, default=-30.0)
    parser.add_argument('--center_x_max_real', type=float, default=30.0)
    parser.add_argument('--center_y_min_real', type=float, default=-30.0)
    parser.add_argument('--center_y_max_real', type=float, default=30.0)

    parser.add_argument('--min_radius_real', type=float, default=3.0)
    parser.add_argument('--max_radius_real', type=float, default=10.0)

    parser.add_argument('--min_speed', type=float, default=0.5)
    parser.add_argument('--max_speed', type=float, default=3.0)

    parser.add_argument('--min_curvature', type=float, default=0.0)
    parser.add_argument('--max_curvature', type=float, default=0.2)

    parser.add_argument('--only_circles', default=True, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--history', type=int, default=1)
    parser.add_argument('--include_start_state', default=False, action='store_true')

    args = parser.parse_args()

    # Instantiate your env wrapper (which holds the new Gym env internally)
    env = Traj2D_Environment(
        scale=args.env_scale,
        history=args.history,
        include_start_state=args.include_start_state,
        visualize_reward=False,  # faster generation
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        # ensure your wrapper passes through max_delta_theta if configurable
    )

    # Generate and save the episodes
    generate_and_save(env, args)