import os
import math
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from functools import partial

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

XML_FILES = {
    'HalfCheetah-Mujoco': 'half_cheetah.xml',
    'HalfCheetah-OGBench': os.path.join(os.path.dirname(__file__), 'assets', 'cheetah.xml')
}

def _gaussian_reward(val, target, sigma):
    """A Gaussian reward function that returns 1 at the target and decays."""
    return float(np.exp(-0.5 * ((val - target) / (sigma + 1e-8)) ** 2))

def _linear_reward(val, target):
    """A linear reward function that clips at 1, rewarding values up to a target."""
    return float(np.clip(val / (target + 1e-8), 0, 1))

class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    """
    A standalone, styled version of the Half-Cheetah environment built from the
    base MujocoEnv. It includes a flexible reward system to encourage various
    behavioral archetypes.
    """

    def __init__(
        self,
        xml_file: str = 'HalfCheetah-Mujoco',
        frame_skip: int = 5,
        archetype: str = "Expert",
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        render_modes: list[str] = ['human', 'rgb_array', 'depth_array', 'rgbd_tuple'],
        render_mode: str = 'rgb_array',
        render_fps: int = 20,
        width: int = 700,
        height: int = 700,
        exclude_current_positions_from_observation: bool = True,
        target_speed: float = 10.0, 
        target_direction: int = 1,
        target_angle: float = 0.0, 
        sigma_angle: float = 0.05,
        target_height: float = 0.6, 
        sigma_height: float = 0.04,
        **kwargs,
    ):
        # Define xml_file
        assert xml_file in XML_FILES
        xml_file = XML_FILES[xml_file]

        # Set reward function
        self.reward_function_map = {
            # Standard reward
            "Expert": self._reward_expert,
            # Base Locomotion
            "Speed_Slow": partial(self._reward_speed, target_speed=1.5),
            "Speed_Medium": partial(self._reward_speed, target_speed=5.0),
            "Speed_Fast": partial(self._reward_speed, target_speed=10.0),
            "Direction_Left": partial(self._reward_direction, target_direction=-1),
            "Direction_Right": partial(self._reward_direction, target_direction=1),
            # Body position
            "Angle_Upright": partial(self._reward_angle, target_angle=-0.2, sigma_angle=0.05),
            "Angle_Flat": partial(self._reward_angle, target_angle=0.0, sigma_angle=0.05),
            "Angle_Crouching": partial(self._reward_angle, target_angle=0.2, sigma_angle=0.05),
            "Height_Running": partial(self._reward_height, target_height=0.7, sigma_height=0.04),
            "Height_Normal": partial(self._reward_height, target_height=0.6, sigma_height=0.04),
            "Height_Crawling": partial(self._reward_height, target_height=0.5, sigma_height=0.04),
            # Other (not fixed) archetypes
            "Frequency_High": partial(self._reward_frequency, target_freq=5.0, sigma_freq=0.5),
            "Frequency_Low": partial(self._reward_frequency, target_freq=1.5, sigma_freq=0.5),
            "Gait_Bounding": partial(self._reward_gait, gait_style="bound"),
            "Gait_Trotting": partial(self._reward_gait, gait_style="trot"),
            "Energy_High": partial(self._reward_energy, target_energy=100.0),
            "Energy_Low": self._reward_energy_low,
            "Stability_Stable": partial(self._reward_stability, stable=True, sigma=1.0),
            "Stability_Unstable": partial(self._reward_stability, stable=False, sigma=1.0),
            # Mix archetypes
            "Mix": partial(self._reward_mix,
                target_speed=target_speed, target_direction=target_direction,
                target_angle=target_angle, sigma_angle=sigma_angle,
                target_height=target_height, sigma_height=sigma_height,
            )
        }
        if archetype not in self.reward_function_map:
            raise ValueError(f"Unknown archetype: {archetype}. Available archetypes are: {list(self.reward_function_map.keys())}")
        self.reward_fn = self.reward_function_map[archetype]
        self.archetype = archetype

        # Initialize environment and utils
        utils.EzPickle.__init__(
            self,
            xml_file=xml_file,
            frame_skip=frame_skip,
            archetype=archetype,
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._time_since_last_bfoot_land = 0.0
        self._bfoot_contact = False
        self._ffoot_contact = False

        self.metadata = {
            "render_modes": render_modes,
            "render_fps": render_fps
        }
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=width,
            height=height,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            camera_id=0,
            **kwargs,
        )
        self.metadata = {
            "render_modes": render_modes,
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            (self.data.qpos.size - 1) if self._exclude_current_positions_from_observation
            else self.data.qpos.size
        ) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _compute_metrics(self):
        """Computes all physical metrics once per step."""
        metrics = {}
        # Base kinematics
        metrics["x_velocity"] = (self.data.qpos[0] - self._x_position_before) / self.dt
        metrics["torso_z"] = self.body_z("torso")
        metrics["torso_angle"] = self.body_angle("torso")
        
        # Frequency
        self._time_since_last_bfoot_land += self.dt
        is_new_contact = self.body_lowest_geom_bottom_z("bfoot") < 0.03 and not self._bfoot_contact
        if is_new_contact:
            # Only record frequency if it's a plausible step, not a bounce
            if self._time_since_last_bfoot_land > 0.05:
                metrics["step_freq"] = 1.0 / (self._time_since_last_bfoot_land)
            else:
                metrics["step_freq"] = 0.0
            self._time_since_last_bfoot_land = 0.0
        else:
            metrics["step_freq"] = 0.0 # No reward if no new step
        self._bfoot_contact = self.body_lowest_geom_bottom_z("bfoot") < 0.03

        # Gait and Stability
        bfoot_z = self.body_lowest_geom_bottom_z("bfoot")
        ffoot_z = self.body_lowest_geom_bottom_z("ffoot")
        metrics["bfoot_z"] = bfoot_z
        metrics["ffoot_z"] = ffoot_z
        metrics["is_airborne"] = (bfoot_z > 0.03) and (ffoot_z > 0.03)
        metrics["thigh_angle_diff"] = self.data.qpos[2] - self.data.qpos[5] # bthigh - fthigh
        metrics["torso_roll"] = self.body_roll("torso")
        metrics["lateral_velocity"] = self.data.qvel[1]
        
        # Energy
        metrics["joint_energy"] = float(np.sum(np.square(self.data.qvel)))
        
        return metrics

    def step(self, action):
        self._x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)

        metrics = self._compute_metrics()
        archetype_reward, reward_info = self.reward_fn(metrics, action)

        observation = self._get_obs()
        info = {
            "x_position": self.data.qpos[0],
            "reward_archetype": archetype_reward,
            **reward_info,
            **metrics, # Add all computed metrics to info for easy logging
        }
        if self.render_mode == "human": self.render()
        return observation, archetype_reward, False, False, info

    # -------------------------------------------------------------------- #
    # ---------- INDIVIDUAL REWARD FUNCTIONS FOR EACH ARCHETYPE ---------- #
    # -------------------------------------------------------------------- #
    def _reward_expert(self, metrics: dict, action: np.ndarray):
        """
        [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
        """
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        reward = forward_reward - ctrl_cost
        return reward, {
            "forward_reward": forward_reward, 
            "ctrl_cost": ctrl_cost
        }

    def _reward_speed(self, metrics: dict, action: np.ndarray, target_speed: float):
        """
        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
        """
        ctrl_cost = self.control_cost(action)
        speed_reward = -abs(metrics["x_velocity"] - target_speed)
        reward = speed_reward - ctrl_cost
        return reward, {
            "speed_reward": speed_reward, 
            "ctrl_cost": ctrl_cost
        }
    
    def _reward_direction(self, metrics: dict, action: np.ndarray, target_direction: int):
        """
        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
        """
        ctrl_cost = self.control_cost(action)
        direction_reward = target_direction * self._forward_reward_weight * metrics["x_velocity"]
        reward = direction_reward - ctrl_cost
        return reward, {
            "direction_reward": direction_reward, 
            "ctrl_cost": ctrl_cost
        }

    def _reward_angle(self, metrics: dict, action: np.ndarray, target_angle: float, sigma_angle: float):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        angle_reward = _gaussian_reward(metrics["torso_angle"], target_angle, sigma_angle)
        reward = angle_reward + angle_reward * forward_reward - ctrl_cost
        return reward, {
            "angle_reward": angle_reward, 
            "forward_reward": forward_reward, 
            "ctrl_cost": ctrl_cost
        }

    def _reward_height(self, metrics: dict, action: np.ndarray, target_height: float, sigma_height: float):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        height_reward = _gaussian_reward(metrics["torso_z"], target_height, sigma_height)
        reward = height_reward + height_reward * forward_reward - ctrl_cost
        return reward, {
            "height_reward": height_reward, 
            "forward_reward": forward_reward, 
            "ctrl_cost": ctrl_cost
        }

    def _reward_frequency(self, metrics: dict, action: np.ndarray, target_freq: float, sigma_freq: float):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        # Only apply reward if a step was actually taken in this frame
        freq_reward = 0.0
        if metrics["step_freq"] > 0:
            freq_reward = _gaussian_reward(metrics["step_freq"], target_freq, sigma_freq)
        reward = freq_reward + freq_reward * forward_reward - ctrl_cost
        return reward, {
            "reward_freq": freq_reward, 
            "reward_forward": forward_reward, 
            "cost_ctrl": -ctrl_cost
        }

    def _reward_gait(self, metrics: dict, action: np.ndarray, gait_style: str):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        
        symmetry_reward = _gaussian_reward(metrics["thigh_angle_diff"], 0.0, sigma=0.1)
        air_time_reward = 1.0 if metrics["is_airborne"] else 0.0
        
        gait_reward = 0.0
        if gait_style == "bound":
            # Bounding is symmetric with significant air time
            gait_reward = symmetry_reward * (0.5 + 0.5 * air_time_reward)
        elif gait_style == "trot":
            # Trotting is asymmetric (legs out of phase)
            gait_reward = 1.0 - symmetry_reward
        
        reward = gait_reward + gait_reward * forward_reward - ctrl_cost
        return reward, {
            "reward_gait": gait_reward, 
            "reward_forward": forward_reward, 
            "cost_ctrl": -ctrl_cost
        }
        
    def _reward_energy(self, metrics: dict, action: np.ndarray, target_energy: float):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        energy_reward = _linear_reward(metrics["joint_energy"], target_energy)
        reward = energy_reward + energy_reward * forward_reward - ctrl_cost
        return reward, {
            "reward_energy": energy_reward, 
            "reward_forward": forward_reward, 
            "cost_ctrl": -ctrl_cost
        }

    def _reward_energy_low(self, metrics: dict, action: np.ndarray):
        # This style is primarily about minimizing control effort.
        # We use a much higher weight for the control cost.
        heavy_ctrl_cost = 1.0 * np.sum(np.square(action))
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        reward = forward_reward - heavy_ctrl_cost
        return reward, {
            "reward_forward": forward_reward, 
            "cost_ctrl": -heavy_ctrl_cost
        }

    def _reward_stability(self, metrics: dict, action: np.ndarray, stable: bool, sigma: float):
        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * metrics["x_velocity"]
        
        instability_metric = abs(metrics["torso_roll"]) + abs(metrics["lateral_velocity"])
        stability_score = math.exp(-instability_metric / sigma)
        
        style_reward = stability_score if stable else (1.0 - stability_score)
        
        reward = style_reward + style_reward * forward_reward - ctrl_cost
        return reward, {
            "reward_stability": style_reward, 
            "reward_forward": forward_reward, 
            "cost_ctrl": -ctrl_cost
        }
    
    def _reward_mix(
            self, 
            metrics: dict, action: np.ndarray,
            target_speed: float, target_direction: int,
            target_angle: float, sigma_angle: float,
            target_height: float, sigma_height: float
        ):
            ctrl_cost = self.control_cost(action)
            direction_speed_reward = -abs(metrics["x_velocity"] - target_direction * target_speed)

            angle_reward = _gaussian_reward(metrics["torso_angle"], target_angle, sigma_angle)
            height_reward = _gaussian_reward(metrics["torso_z"], target_height, sigma_height)
            position_reward = angle_reward * height_reward

            reward = position_reward + position_reward * direction_speed_reward - ctrl_cost

            return reward, {
                "angle_reward": angle_reward,
                "height_reward": height_reward, 
                "direction_speed_reward": direction_speed_reward, 
                "ctrl_cost": ctrl_cost
            }

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity))

    def reset_model(self):
        self._time_since_last_bfoot_land = 0.0
        self._time_since_last_ffoot_land = 0.0
        self._bfoot_contact = False
        self._ffoot_contact = False
        
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(
            self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def reset(self, seed: int = None, *, options: dict = None):
        obs, infos = super().reset(seed=seed, options=options)

        metrics = {}
        # Base kinematics
        metrics["x_velocity"] = 0.0
        metrics["torso_z"] = self.body_z("torso")
        metrics["torso_angle"] = self.body_angle("torso")
        # Frequency
        metrics["step_freq"] = 0.0
        # Gait and Stability
        bfoot_z = self.body_lowest_geom_bottom_z("bfoot")
        ffoot_z = self.body_lowest_geom_bottom_z("ffoot")
        metrics["bfoot_z"] = bfoot_z
        metrics["ffoot_z"] = ffoot_z
        metrics["is_airborne"] = (bfoot_z > 0.03) and (ffoot_z > 0.03)
        metrics["thigh_angle_diff"] = self.data.qpos[2] - self.data.qpos[5] # bthigh - fthigh
        metrics["torso_roll"] = self.body_roll("torso")
        metrics["lateral_velocity"] = self.data.qvel[1]
        # Energy
        metrics["joint_energy"] = float(np.sum(np.square(self.data.qvel)))
        # Additional metrics
        reward_infos = {
            'reward_archetype': 0.0,
            'forward_reward': 0.0,
            'ctrl_cost': 0.0,

        }
        infos.update({
            "x_position": self.data.qpos[0],
            **reward_infos,
            **metrics, # Add all computed metrics to info for easy logging
        })
        return obs, infos

    def body_z(self, body_name: str) -> float:
        return float(self.data.body(body_name).xpos[2])

    def body_angle(self, body_name: str) -> float:
        xmat = self.data.body(body_name).xmat.reshape(3, 3)
        return math.atan2(-xmat[2, 0], xmat[0, 0])
        
    def body_roll(self, body_name: str) -> float:
        xmat = self.data.body(body_name).xmat.reshape(3, 3)
        return math.atan2(xmat[2, 1], xmat[2, 2])

    def body_lowest_geom_bottom_z(self, body_name: str) -> float:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        start, count = int(self.model.body_geomadr[bid]), int(self.model.body_geomnum[bid])
        if count == 0: return self.body_z(body_name)
        zmins = [self._geom_bottom_z(gi) for gi in range(start, start + count)]
        return float(np.min(zmins))

    def _geom_bottom_z(self, gi: int) -> float:
        xpos, xmat, size = self.data.geom_xpos[gi], self.data.geom_xmat[gi].reshape(3, 3), self.model.geom_size[gi]
        gtype = int(self.model.geom_type[gi])
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE: return float(xpos[2] - size[0])
        if gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            end1, end2 = xpos - xmat[:, 2] * size[1], xpos + xmat[:, 2] * size[1]
            return float(min(end1[2], end2[2]) - size[0])
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            zproj = abs(xmat[2, 0])*size[0] + abs(xmat[2, 1])*size[1] + abs(xmat[2, 2])*size[2]
            return float(xpos[2] - zproj)
        return float(xpos[2])

if __name__ == '__main__':

    ARCHETYPE_LIST = list(HalfCheetahEnv(archetype="Expert").reward_function_map.keys())

    def run_style(style_name, steps=1000):
        print(f"\n--- Running Archetype: {style_name} ---")
        env = HalfCheetahEnv(xml_file='HalfCheetah-Mujoco', render_mode="human", archetype=style_name)
        obs, info = env.reset()
        for i in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 200 == 0:
                print(f"Step {i}: {info}")
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    for name in ARCHETYPE_LIST:
        run_style(name)