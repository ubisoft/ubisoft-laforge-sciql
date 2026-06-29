import os
import math
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from typing import Union, Optional
from functools import partial


XML_FILES = {
    'Ant-Mujoco': 'ant.xml',
    'Ant-OGBench': os.path.join(os.path.dirname(__file__), 'assets', 'ant.xml')
}

def _gaussian_reward(val, target, sigma):
    """A Gaussian reward function that returns 1 at the target and decays."""
    return float(np.exp(-0.5 * ((val - target) / (sigma + 1e-8)) ** 2))

class AntEnv(MujocoEnv, utils.EzPickle):
    """Gymnasium Stylized Ant environment.

    There are two types of Ant envrionments in the literature. The Ant-Mujoco for locomotion tasks (go as fast as possible),
    and the navigation Ant-OGBench for navigation in a maze. Each have a different xml file as well as different behaviors.
    You can retrieve the behaviors for each envs by using the following parameters:
        For getting the OGBench Ant:
            xml_file: str = 'Ant-OGBench',
            archetype: str = "Directional_Expert",
            forward_reward_weight: float = 1.0,
            healthy_reward_weight: float = 0.0,
            ctrl_cost_weight: float = 0.0,
            contact_cost_weight: float = 0.0,
            terminate_when_unhealthy: bool = False,
            exclude_current_positions_from_observation: bool = True,
            include_cfrc_ext_in_observation: bool = False,
            render_fps: int = 10,
        For getting the Mujoco Ant:
            xml_file: str = 'Ant-Mujoco',
            archetype: str = "Expert",
            forward_reward_weight: float = 1.0,
            healthy_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 0.5,
            contact_cost_weight: float = 5e-4,
            terminate_when_unhealthy: bool = True,
            exclude_current_positions_from_observation: bool = True,
            include_cfrc_ext_in_observation: bool = True,
            render_fps: int = 20,
    The two xml files are compatible with each env.
    """

    def __init__(
        self,
        xml_file: str = 'Ant-Mujoco',
        frame_skip: int = 5,
        archetype: str = "Expert",
        forward_reward_weight: float = 1.0,
        healthy_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        contact_force_range: tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        render_modes: list[str] = ['human', 'rgb_array', 'depth_array', 'rgbd_tuple'],
        render_mode: str = 'rgb_array',
        render_fps: int = 20,
        width: int = 700,
        height: int = 700,
        camera_name: str = 'back',
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        target_speed: float = 3.0, # Recommanded: 1.0, 2.0, 3.0
        target_height: float = 0.6, # Recommanded: 0.4, 0.6, 0.8
        target_yaw_deg: float = 0.0, # Recommanded: 0.0, 180, 90, -90
        resample_interval: int = 100,
        **kwargs,
    ):
        """Initialize the Ant environment.

        Args:
            xml_file: Path to the XML description (optional).
            reset_noise_scale: Scale of the noise added to the initial state during reset.
            render_mode: Rendering mode.
            width: Width of the rendered image.
            height: Height of the rendered image.
            **kwargs: Additional keyword arguments.
        """
        # Define xml_file
        assert xml_file in XML_FILES
        xml_file = XML_FILES[xml_file]

        # Set reward function
        self._directional = 'Directional' in archetype
        if self._directional:
            self.reward_function_map = {
                # Standard Task Focused Style
                "Directional_Expert": self._reward_directional_expert,
                # Velocity-based Styles
                "Directional_Speed_Slow": partial(self._reward_directional_speed, target_speed=1.0),
                "Directional_Speed_Medium": partial(self._reward_directional_speed, target_speed=2.0),
                "Directional_Speed_Fast": partial(self._reward_directional_speed, target_speed=3.0),
                # Body Orientation Styles
                "Directional_Orientation_Forward": partial(self._reward_directional_orientation, target_yaw_deg=0.0),
                "Directional_Orientation_Backward": partial(self._reward_directional_orientation, target_yaw_deg=180),
                "Directional_Orientation_Left": partial(self._reward_directional_orientation, target_yaw_deg=90),
                "Directional_Orientation_Right": partial(self._reward_directional_orientation, target_yaw_deg=-90),
                # Height Styles
                "Directional_Height_Crawl": partial(self._reward_directional_height, target_height=0.4),
                "Directional_Height_Normal": partial(self._reward_directional_height, target_height=0.6),
                "Directional_Height_Upright": partial(self._reward_directional_height, target_height=0.8),
                # Mix Style
                "Directional_Mix": partial(
                    self._reward_directional_mix,
                    target_speed=target_speed,
                    target_height=target_height, 
                    target_yaw_deg=target_yaw_deg
                )
            }
            self.z = None
            self.num_steps = 0
            self.resample_interval = resample_interval
        else:
            self.reward_function_map = {
                # Standard Task Focused Style
                "Expert": self._reward_expert,
                # Velocity-based Styles
                "Speed_Slow": partial(self._reward_speed, target_speed=1.0),
                "Speed_Medium": partial(self._reward_speed, target_speed=2.0),
                "Speed_Fast": partial(self._reward_speed, target_speed=3.0),
                # Directional Styles
                "Direction_Forward": partial(self._reward_direction, target_direction_deg=0.0),
                "Direction_Backward": partial(self._reward_direction, target_direction_deg=180),
                "Direction_Left": partial(self._reward_direction, target_direction_deg=90),
                "Direction_Right": partial(self._reward_direction, target_direction_deg=-90),
                # Body Orientation Styles
                "Orientation_Forward": partial(self._reward_orientation, target_yaw_rad=0.0),
                "Orientation_Backward": partial(self._reward_orientation, target_yaw_rad=180),
                "Orientation_Left": partial(self._reward_orientation, target_yaw_rad=90),
                "Orientation_Right": partial(self._reward_orientation, target_yaw_rad=-90),
                # Height Styles
                "Height_Crawl": partial(self._reward_height, target_height=0.4),
                "Height_Normal": partial(self._reward_height, target_height=0.6),
                "Height_Upright": partial(self._reward_height, target_height=0.8),
                # Gait Styles
                "Gait_Trot": partial(self._reward_gait, gait_style="trot"),
                "Gait_Pace": partial(self._reward_gait, gait_style="pace"),
                "Gait_Bound": partial(self._reward_gait, gait_style="bound"),
                # Stability Style
                "Stability_Stable": partial(self._reward_stability, stability_sigma=0.5),
                # Mix Style
                "Mix": partial(
                    self._reward_mix,
                    target_speed=target_speed,
                    target_height=target_height, 
                    target_yaw_deg=target_yaw_deg
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
            contact_cost_weight=contact_cost_weight, 
            healthy_reward=healthy_reward,
            main_body=main_body,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            contact_force_range=contact_force_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation=include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._healthy_reward_weight = healthy_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._main_body = main_body
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

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
            camera_name=camera_name,
            **kwargs,
        )
        self.metadata = {
            "render_modes": render_modes,
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        obs_size += 2 * self._directional

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    @property
    def healthy_reward(self):
        return self._healthy_reward_weight * self.is_healthy * self._healthy_reward
    
    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))
    
    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost
    
    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
    
    def _quaternion_to_euler(self, quaternion):
        w, x, y, z = quaternion
        roll_x = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch_y = math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0)) # Clip to avoid domain error
        yaw_z = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return roll_x, pitch_y, yaw_z
    
    def _compute_metrics(self, action):
        metrics = {}
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()
        xy_velocity = (xy_position_after - self._xy_position_before) / self.dt
        metrics["xy_position_before"] = self._xy_position_before
        metrics["xy_position_after"] = xy_position_after
        metrics["xy_velocity"] = xy_velocity
        metrics["speed"] = np.linalg.norm(xy_velocity)
        metrics["direction"] = math.atan2(xy_velocity[1], xy_velocity[0] + 1e-8)
        
        metrics["torso_z"] = self.data.qpos[2]
        torso_quat = self.data.body('torso').xquat
        metrics["torso_roll"], metrics["torso_pitch"], metrics["torso_yaw"] = self._quaternion_to_euler(torso_quat)


        metrics["is_healthy"] = self.is_healthy
        metrics["healthy_reward"] = self.healthy_reward

        metrics["ctrl_cost"] = self.control_cost(action)
        metrics["contact_cost"] = self.contact_cost
        
        foot_contacts = [False] * 4
        foot_geom_names = ["front_left_foot", "front_right_foot", "back_left_foot", "right_back_foot"]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if "floor" in geom1_name or "floor" in geom2_name:
                for j, foot_name in enumerate(foot_geom_names):
                    if foot_name in geom1_name or foot_name in geom2_name:
                        foot_contacts[j] = True
        metrics["foot_contacts"] = foot_contacts
        
        return metrics
    
    def step(self, action):

        self._xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        prev_qpos = self.data.qpos.copy()
        prev_qvel = self.data.qvel.copy()

        self.do_simulation(action, self.frame_skip)

        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        metrics = self._compute_metrics(action)
        archetype_reward, reward_info = self.reward_fn(metrics)
        terminated = (not metrics["is_healthy"]) and self._terminate_when_unhealthy
        observation = self._get_obs()

        info = {
            'reward_archetype': archetype_reward,
            'prev_qpos': prev_qpos,
            'prev_qvel': prev_qvel,
            'qpos': qpos,
            'qvel': qvel,
            **reward_info,
            **metrics,
        }

        if self._directional:
            self.num_steps += 1
            if self.num_steps % self.resample_interval == 0:
                self.z = np.random.randn(2)
                self.z = self.z / np.linalg.norm(self.z)
            info["xy"] = metrics["xy_position_after"]
            info["direction"] = self.z
            observation = np.concatenate([observation, self.z])

        if self.render_mode == "human": self.render()

        return observation, archetype_reward, terminated, False, info

    # -------------------------------------------------------------------- #
    # ---------- INDIVIDUAL REWARD FUNCTIONS FOR EACH ARCHETYPE ---------- #
    # -------------------------------------------------------------------- #
    def _reward_expert(self, metrics: dict):
        """
        [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
        """
        # Forward reward
        forward_reward = self._forward_reward_weight * metrics["xy_velocity"][0]
        # Total reward
        reward = forward_reward + metrics["healthy_reward"] - metrics["ctrl_cost"] - metrics["contact_cost"]
        return reward, {
            "forward_reward": forward_reward, 
            "healthy_reward": metrics["healthy_reward"], 
            "ctrl_cost": -metrics["ctrl_cost"], 
            "contact_cost": -metrics["contact_cost"]
        }

    def _reward_directional_expert(self, metrics: dict):
        # Direction reward
        direction_reward = (metrics["xy_position_after"] - metrics["xy_position_before"]).dot(self.z)
        # Total reward
        reward = direction_reward + metrics["healthy_reward"] - metrics["ctrl_cost"] - metrics["contact_cost"]
        return reward, {
            "direction_reward": direction_reward, 
            "healthy_reward": metrics["healthy_reward"], 
            "ctrl_cost": -metrics["ctrl_cost"], 
            "contact_cost": -metrics["contact_cost"],
        }

    def _reward_speed(self, metrics: dict, target_speed: float):
        """
        [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
        """
        # Speed reward
        speed_reward = -abs(metrics["xy_velocity"][0] - target_speed)
        # Total reward
        reward = speed_reward + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "reward_speed": speed_reward, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }

    def _reward_directional_speed(self, metrics: dict, target_speed: float):
        # Direction reward
        displacement = metrics["xy_position_after"] - metrics["xy_position_before"]
        normalized_displacement = displacement / np.linalg.norm(displacement)
        direction_reward = (1 + normalized_displacement.dot(self.z)) / 2
        # Speed reward
        speed_reward = -abs(metrics["speed"] - target_speed)
        # Total reward
        reward = direction_reward * speed_reward + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "reward_speed": speed_reward, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }

    def _reward_direction(self, metrics: dict, target_direction_deg: float):
        # Direction reward
        target_direction_rad = target_direction_deg * np.pi / 180
        angle_diff = (metrics["direction"] - target_direction_rad + np.pi) % (2 * np.pi) - np.pi
        direction_reward = _gaussian_reward(angle_diff, 0.0, sigma=0.25)
        # Total reward
        reward = (direction_reward * metrics["speed"]) + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "reward_direction": direction_reward, 
            "speed": metrics["speed"], 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }
    
    def _reward_orientation(self, metrics: dict, target_yaw_deg: float):
        """
        Calculates a reward for moving at a specific angle relative to body orientation.

        Args:
            metrics (dict): A dictionary containing simulation metrics.
                            Requires "torso_yaw" and "xy_velocity".
            target_yaw_rad (float): The desired angle between the ant's forward
                                    direction and its velocity vector.
                                    - 0.0 means move forward.
                                    - np.pi / 2 means strafe perfectly to the left.
                                    - -np.pi / 2 means strafe perfectly to the right.
                                    - np.pi means move backward.
        """
        # Orientation reward
        velocity_vec = metrics["xy_velocity"]
        speed = np.linalg.norm(velocity_vec)
        if speed < 0.1:
            orientation_reward = 0
            actual_angle_rad = 0
        else:
            # 1. Get the ant's forward-facing direction as a 2D unit vector.
            torso_yaw = metrics["torso_yaw"]
            orientation_vec = np.array([np.cos(torso_yaw), np.sin(torso_yaw)])
            forward_speed = np.dot(velocity_vec, orientation_vec)
            
            # Create a sideways vector (90 degrees left of forward)
            sideways_vec = np.array([-np.sin(torso_yaw), np.cos(torso_yaw)])
            sideways_speed = np.dot(velocity_vec, sideways_vec)

            # 3. Calculate the actual angle of movement relative to the body.
            #    arctan2 gives the full signed angle from -pi to pi.
            actual_angle_rad = np.arctan2(sideways_speed, forward_speed)
            
            # 4. Calculate the error between the actual angle and the target angle.
            #    We must wrap this difference to the [-pi, pi] interval.
            target_yaw_rad = target_yaw_deg * np.pi / 180
            angle_error = actual_angle_rad - target_yaw_rad
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
            
            # 5. Convert the error into a reward using a Gaussian function.
            #    This gives a reward of 1 for a perfect angle, decaying as error increases.
            orientation_reward = _gaussian_reward(angle_error, 0.0, sigma=0.5) # Wider sigma is more permissive
        
        # Total reward
        # 6. The final reward should encourage both correct alignment AND speed. Combine with other reward components.
        reward = orientation_reward * metrics["xy_velocity"][0] + metrics["healthy_reward"] - metrics["ctrl_cost"]

        return reward, {
            "orientation_reward": orientation_reward,
            "reward_survive": metrics["healthy_reward"],
            "cost_ctrl": -metrics["ctrl_cost"],
            "actual_relative_angle_deg": np.rad2deg(actual_angle_rad), # For logging
        }
    
    def _reward_directional_orientation(self, metrics: dict, target_yaw_deg: float):
        # Direction reward
        direction_reward = (metrics["xy_position_after"] - metrics["xy_position_before"]).dot(self.z)
        # Orientation reward
        velocity_vec = metrics["xy_velocity"]
        speed = np.linalg.norm(velocity_vec)
        if speed < 0.1:
            orientation_reward = 0
        else:
            torso_yaw = metrics["torso_yaw"]
            orientation_vec = np.array([np.cos(torso_yaw), np.sin(torso_yaw)])
            forward_speed = np.dot(velocity_vec, orientation_vec)
            sideways_vec = np.array([-np.sin(torso_yaw), np.cos(torso_yaw)])
            sideways_speed = np.dot(velocity_vec, sideways_vec)
            actual_angle_rad = np.arctan2(sideways_speed, forward_speed)
            target_yaw_rad = target_yaw_deg * np.pi / 180
            angle_error = actual_angle_rad - target_yaw_rad
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
            orientation_reward = _gaussian_reward(angle_error, 0.0, sigma=0.5)
        # Total reward
        reward = orientation_reward * direction_reward + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "reward_orientation": orientation_reward, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }

    def _reward_height(self, metrics: dict, target_height: float):
        # Height reward
        height_reward = _gaussian_reward(metrics["torso_z"], target_height, sigma=0.1)
        # Expert reward
        forward_progress = self._forward_reward_weight * metrics["xy_velocity"][0]
        # Total reward
        reward = (height_reward * forward_progress) + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "reward_height": height_reward, 
            "reward_forward": forward_progress, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }
    
    def _reward_directional_height(self, metrics: dict, target_height: float):
        # Direction reward
        direction_reward = (metrics["xy_position_after"] - metrics["xy_position_before"]).dot(self.z)
        # Height reward
        height_reward = _gaussian_reward(metrics["torso_z"], target_height, sigma=0.1)
        # Total reward
        reward = height_reward * direction_reward + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "reward_height": height_reward, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }

    def _reward_gait(self, metrics: dict, gait_style: str):
        foot_contacts = metrics["foot_contacts"]
        gait_reward = 0.0
        if gait_style == "trot":
            diag1_sync = float(foot_contacts[0] == foot_contacts[3])
            diag2_sync = float(foot_contacts[1] == foot_contacts[2])
            gait_reward = (diag1_sync + diag2_sync) / 2.0
        elif gait_style == "pace":
            left_sync = float(foot_contacts[0] == foot_contacts[2])
            right_sync = float(foot_contacts[1] == foot_contacts[3])
            gait_reward = (left_sync + right_sync) / 2.0
        elif gait_style == "bound":
            front_sync = float(foot_contacts[0] == foot_contacts[1])
            back_sync = float(foot_contacts[2] == foot_contacts[3])
            gait_reward = (front_sync + back_sync) / 2.0
        forward_progress = self._forward_reward_weight * metrics["xy_velocity"][0]
        reward = gait_reward + gait_reward * forward_progress + metrics['healthy_reward'] - metrics["ctrl_cost"]
        return reward, {"reward_gait": gait_reward, "reward_forward": forward_progress, "cost_ctrl": -metrics["ctrl_cost"]}

    def _reward_stability(self, metrics: dict, stability_sigma: float):
        # Stability reward
        instability_metric = abs(metrics["torso_roll"]) + abs(metrics["torso_pitch"])
        stability_reward = math.exp(-instability_metric / stability_sigma)
        forward_progress = self._forward_reward_weight * metrics["xy_velocity"][0]
        # Total reward
        reward = (stability_reward * forward_progress) + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {"reward_stability": stability_reward, "reward_forward": forward_progress, "reward_survive": metrics["healthy_reward"], "cost_ctrl": -metrics["ctrl_cost"]}

    def _reward_mix(self, metrics: dict, target_speed: float, target_height: float, target_yaw_deg: float):
        # Speed reward
        speed_reward = -abs(metrics["xy_velocity"][0] - target_speed)
        # Orientation reward
        velocity_vec = metrics["xy_velocity"]
        speed = np.linalg.norm(velocity_vec)
        if speed < 0.1:
            orientation_reward = 0
        else:
            torso_yaw = metrics["torso_yaw"]
            orientation_vec = np.array([np.cos(torso_yaw), np.sin(torso_yaw)])
            forward_speed = np.dot(velocity_vec, orientation_vec)
            sideways_vec = np.array([-np.sin(torso_yaw), np.cos(torso_yaw)])
            sideways_speed = np.dot(velocity_vec, sideways_vec)
            actual_angle_rad = np.arctan2(sideways_speed, forward_speed)
            target_yaw_rad = target_yaw_deg * np.pi / 180
            angle_error = actual_angle_rad - target_yaw_rad
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
            orientation_reward = _gaussian_reward(angle_error, 0.0, sigma=0.5)
        # Height reward
        height_reward = _gaussian_reward(metrics["torso_z"], target_height, sigma=0.1)
        # Mix reward
        reward = (orientation_reward * height_reward * speed_reward) + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "orientation_reward": orientation_reward,
            "height_reward": height_reward, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }

    def _reward_directional_mix(self, metrics: dict, target_speed: float, target_height: float, target_yaw_deg: float):
        # Speed reward
        speed_reward = -abs(metrics["speed"] - target_speed)
        # Direction reward
        displacement = metrics["xy_position_after"] - metrics["xy_position_before"]
        normalized_displacement = displacement / np.linalg.norm(displacement)
        direction_reward = (1 + normalized_displacement.dot(self.z)) / 2
        # Orientation reward
        velocity_vec = metrics["xy_velocity"]
        speed = np.linalg.norm(velocity_vec)
        if speed < 0.1:
            orientation_reward = 0
        else:
            torso_yaw = metrics["torso_yaw"]
            orientation_vec = np.array([np.cos(torso_yaw), np.sin(torso_yaw)])
            forward_speed = np.dot(velocity_vec, orientation_vec)
            sideways_vec = np.array([-np.sin(torso_yaw), np.cos(torso_yaw)])
            sideways_speed = np.dot(velocity_vec, sideways_vec)
            actual_angle_rad = np.arctan2(sideways_speed, forward_speed)
            target_yaw_rad = target_yaw_deg * np.pi / 180
            angle_error = actual_angle_rad - target_yaw_rad
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
            orientation_reward = _gaussian_reward(angle_error, 0.0, sigma=0.5)
        # Height reward
        height_reward = _gaussian_reward(metrics["torso_z"], target_height, sigma=0.1)
        # Total reward
        reward = (orientation_reward * height_reward * direction_reward * speed_reward) + metrics["healthy_reward"] - metrics["ctrl_cost"]
        return reward, {
            "orientation_reward": orientation_reward,
            "height_reward": height_reward, 
            "direction_reward": direction_reward, 
            "reward_survive": metrics["healthy_reward"], 
            "cost_ctrl": -metrics["ctrl_cost"]
        }
        
    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        observation = self.reset_model()
        info = self._get_reset_info()

        if self._directional:
            self.z = np.random.randn(2)
            self.z = self.z / np.linalg.norm(self.z)
            self.num_steps = 0
            observation = np.concatenate([observation, self.z])
            info['direction'] = self.z

        return observation, info

    def reset_model(self):
        noise = self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=self.model.nq)
        qpos = self.init_qpos + noise
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        self._xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        return self._get_obs()

    def _get_reset_info(self):
        return {"x_position": self.data.qpos[0], "y_position": self.data.qpos[1]}

if __name__ == '__main__':

    env = AntEnv(xml_file='Ant-OGBench', render_fps=10, render_mode='human', width=1000, height=1000)
    obs, infos = env.reset()
    while True:
        env.render()
        env.step(env.action_space.sample())