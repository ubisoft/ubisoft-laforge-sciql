import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from collections import deque


class Traj2D(gym.Env[np.ndarray, np.ndarray]):
    """
    2D ring-target environment.

    Action (np.array([dtheta_norm, dx_norm])):
        dtheta_norm in [-1, 1] -> Δθ in [-max_delta_theta, +max_delta_theta] (radians) applied this step
        dx_norm     in [-1, 1] -> step distance along heading:
                                 maps to speed in [min_speed, max_speed] (real units/step),
                                 then Δx_env = speed / scale (env-units distance)

    Observation (flattened):
        History of [x, y, θ]: [x_t, y_t, θ_t, x_{t-1}, y_{t-1}, θ_{t-1}, ...]
        (+ [x0, y0] appended if include_start_state=True)

    Reward (selectable via `reward_type`):
        - "gaussian":  r = exp( - d(s_{t+1})^2 / (2 * sigma^2) ),  d = | ||p-c|| - R |
        - "neg_abs" :  r = - d(s_{t+1})

        (Note: the distance background in render remains Z = d, i.e., distance-to-circle.)

    Rendering:
        - Background colormap shows the distance-to-circle field (Z = |error|).
        - 'fixed' or 'follow' camera.
        - Boundary box and outside shading if limits are enabled.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        num_modes: int = 4,
        scale: float = 1.0,
        has_limits: bool = True,
        limits: list = [[-50.0, 50.0], [-50.0, 50.0]],
        terminates_on_limits: bool = False,
        history: int = 2,
        include_start_state: bool = True,
        # step distance is derived from speed in real units (per step)
        min_speed: float = 0.5,
        max_speed: float = 3.0,
        # turning
        max_delta_theta: float = np.pi,   # max turn per step (radians)
        # distance colormap options
        visualize_reward: bool = True,    # (visualizes distance-to-circle)
        reward_grid_res: int = 220,
        reward_cmap: str = "viridis",
        reward_alpha: float = 0.55,
        reward_contrast: float = 20.0,    # real-units span used as vmax for visualization
        # ---------- NEW: reward selection ----------
        reward_type: str = "gaussian",    # "gaussian" or "neg_abs"
        sigma_reward: float = 5.0,        # Gaussian σ in REAL units
    ):
        super().__init__()

        assert history >= 1, "History must be at least 1."
        self.history = int(history)
        self.include_start_state = bool(include_start_state)

        # ---- Spaces ----
        single_state_low  = np.array([-20.0 / scale, -20.0 / scale, -np.pi], dtype=np.float32)
        single_state_high = np.array([ 20.0 / scale,  20.0 / scale,  np.pi ], dtype=np.float32)

        if self.include_start_state:
            low  = np.concatenate([np.tile(single_state_low,  self.history),
                                   np.array([-20.0 / scale, -20.0 / scale, -np.pi], dtype=np.float32)])
            high = np.concatenate([np.tile(single_state_high, self.history),
                                   np.array([ 20.0 / scale,  20.0 / scale,  np.pi], dtype=np.float32)])
            obs_len = 3 * self.history + 3
        else:
            low  = np.tile(single_state_low,  self.history)
            high = np.tile(single_state_high, self.history)
            obs_len = 3 * self.history

        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_len,), dtype=np.float32)

        # Actions: normalized [-1,1] for (Δθ, step-magnitude control)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.max_delta_theta = float(max_delta_theta)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ---- Scaling and limits ----
        self.scale = float(scale)
        self.has_limits = bool(has_limits)
        self.limits = np.array(limits, dtype=np.float32) / self.scale
        self.terminates_on_limits = bool(terminates_on_limits)

        # For render arrow length (cosmetic)
        self.step_size = 1.0 / self.scale

        # ---- State ----
        # [x, y, theta_state] ; theta_state is observable heading in [-pi, pi]
        self.state: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.theta: float = 0.0
        self.theta_state: float = 0.0
        self._max_episode_steps: int = 1000
        self.t: int = 0

        # ---- Targets (scaled to env units) ----
        self.center_set: np.ndarray = (
            np.array([[0, 5], [-10, -20], [14, -30], [-16, 18]], dtype=np.float32) / self.scale
        )
        self.radius_set: np.ndarray = (
            np.array([5, 10, 5, 10], dtype=np.float32) / self.scale
        )
        assert num_modes <= len(self.radius_set), "num_modes exceeds predefined centers/radii"
        self.num_modes: int = int(num_modes)
        self.mode_idx: int = 0

        # ---- Histories ----
        self.start_pos: Optional[np.ndarray] = None  # (x0, y0) in env units
        self.state_history: deque = deque(maxlen=self.history)  # stores [x, y, θ]
        self.path_history: List[np.ndarray] = []

        # ---- Rendering ----
        self.initialized_ion: bool = False
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

        # ---- Distance map opts (visualization) ----
        self.visualize_reward = bool(visualize_reward)
        self.reward_grid_res = int(reward_grid_res)
        self.reward_cmap = reward_cmap
        self.reward_alpha = float(reward_alpha)
        self.reward_contrast = float(reward_contrast) / self.scale  # env units
        if self.reward_contrast <= 0:
            self.reward_contrast = 1.0

        # small cache for the colormap
        self._rewardmap_cache = {"mode_idx": None, "xlim": None, "ylim": None, "Z": None, "extent": None}

        # ---- Reward selection (NEW) ----
        self.reward_type = str(reward_type).lower()
        if self.reward_type not in ("gaussian", "neg_abs"):
            raise ValueError(f"Unknown reward_type '{reward_type}'. Use 'gaussian' or 'neg_abs'.")
        # Convert σ from REAL units to ENV units
        self.sigma_env = float(sigma_reward) / self.scale
        if self.sigma_env <= 0:
            self.sigma_env = 1e-6  # avoid divide-by-zero

        # ---- Progress reward bookkeeping (kept for info) ----
        self._prev_ring_distance: float = 0.0  # | ||p - c|| - R | at s_t

    # ============================== Core API ==============================

    def _ring_distance(self, pos_xy: np.ndarray, center: np.ndarray, radius: float) -> float:
        """| ||p - center|| - radius |"""
        return float(abs(np.linalg.norm(pos_xy - center) - radius))

    def get_observation(self) -> np.ndarray:
        """Flattened [x,y,θ] history, newest first; optionally appends (x0,y0)."""
        history_list = list(reversed(self.state_history))  # newest first
        hist = np.array(history_list, dtype=np.float32).flatten()
        if self.include_start_state and self.start_pos is not None:
            return np.concatenate((hist, self.start_pos.astype(np.float32)))
        return hist

    def _compute_step_reward(self, cur_dist: float) -> float:
        """Compute step reward based on selected reward_type and current distance."""
        if self.reward_type == "gaussian":
            # r = exp( - d^2 / (2 sigma^2) )
            return float(np.exp(- (cur_dist ** 2) / (2.0 * (self.sigma_env ** 2))))
        elif self.reward_type == "neg_abs":
            # r = - d
            return -float(cur_dist)
        # Should never reach here due to validation in __init__
        return 0.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        terminated, truncated = False, False

        # ---- Map normalized action to physical values (Δθ, step distance) ----
        a_turn = float(np.clip(action[0], -1.0, 1.0))
        a_step = float(np.clip(action[1], -1.0, 1.0))

        delta_theta = a_turn * self.max_delta_theta  # radians
        # map to real-units speed then to env distance for this step
        speed_real = self.min_speed + (a_step + 1.0) * 0.5 * (self.max_speed - self.min_speed)
        step_dist_env = speed_real / self.scale  # distance along heading this step

        # --- distance BEFORE state update (kept in info)
        center = self.center_set[self.mode_idx]
        radius = float(self.radius_set[self.mode_idx])
        prev_dist = self._ring_distance(self.state[:2], center, radius)

        # ---- Kinematics: rotate by Δθ, then move forward by step_dist_env ----
        self.theta = self.state[2] + delta_theta
        # keep observable theta within [-pi, pi]
        self.theta_state = (self.theta + np.pi) % (2 * np.pi) - np.pi

        dx = step_dist_env * np.cos(self.theta_state)
        dy = step_dist_env * np.sin(self.theta_state)

        new_x = self.state[0] + dx
        new_y = self.state[1] + dy

        if self.has_limits:
            inside_next = (self.limits[0, 0] < new_x < self.limits[0, 1]) and (self.limits[1, 0] < new_y < self.limits[1, 1])
            if self.terminates_on_limits and not inside_next:
                terminated = True
            self.state[0] = np.clip(new_x, self.limits[0, 0], self.limits[0, 1])
            self.state[1] = np.clip(new_y, self.limits[1, 0], self.limits[1, 1])
        else:
            self.state[0] = new_x
            self.state[1] = new_y

        self.state[2] = self.theta_state

        # histories
        self.state_history.append(self.state.copy())
        self.path_history.append(self.state[:2].copy())

        # --- distance AFTER state update & reward ---
        cur_dist = self._ring_distance(self.state[:2], center, radius)
        reward = self._compute_step_reward(cur_dist)

        # update memory for next step
        self._prev_ring_distance = cur_dist

        if self.t >= self._max_episode_steps:
            truncated = True

        info = dict(
            true_theta=self.theta_state,
            delta_theta=delta_theta,
            step_distance_env=step_dist_env,
            true_speed=speed_real,  # compatibility with previous tooling
            mode_idx=self.mode_idx,
            ring_distance_prev=prev_dist,
            ring_distance_cur=cur_dist,
            reward_type=self.reward_type,
        )
        return self.get_observation(), float(reward), terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0

        # Defaults (real units before scaling)
        start_x_real = 0.0
        start_y_real = 0.0
        initial_theta = 0.0

        if options:
            # Position selection
            if options.get("random_start_x", True):
                max_x_abs_real = (self.observation_space.high[0] * self.scale) * 0.7
                start_x_real = float(self.np_random.uniform(-max_x_abs_real, max_x_abs_real))
            elif "start_x_real" in options:
                start_x_real = float(options["start_x_real"])

            if options.get("random_start_y", True):
                max_y_abs_real = (self.observation_space.high[1] * self.scale) * 0.7
                start_y_real = float(self.np_random.uniform(-max_y_abs_real, max_y_abs_real))
            elif "start_y_real" in options:
                start_y_real = float(options["start_y_real"])

            # Initial heading
            if options.get("random_initial_theta", True):
                initial_theta = float(self.np_random.uniform(-np.pi, np.pi))
            elif "initial_theta_real" in options:
                initial_theta = float(options["initial_theta_real"])

            # Mode index
            if "mode_idx" in options:
                self.mode_idx = int(options["mode_idx"])
            elif self.num_modes > 0:
                self.mode_idx = int(self.np_random.integers(0, self.num_modes))
            else:
                self.mode_idx = 0
        elif self.num_modes > 0:
            self.mode_idx = int(self.np_random.integers(0, self.num_modes))
        else:
            self.mode_idx = 0

        # State in env units
        self.theta = initial_theta
        self.theta_state = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([start_x_real / self.scale,
                               start_y_real / self.scale,
                               self.theta_state], dtype=np.float32)

        # histories
        self.start_pos = self.state[:3].copy()
        self.state_history.clear()
        for _ in range(self.history):
            self.state_history.append(self.state.copy())
        self.path_history = [self.state[:2].copy()]

        # init previous distance (kept for info)
        center = self.center_set[self.mode_idx]
        radius = float(self.radius_set[self.mode_idx])
        self._prev_ring_distance = self._ring_distance(self.state[:2], center, radius)

        info = dict(
            true_theta=self.theta_state,
            delta_theta=0.0,
            step_distance_env=0.0,
            true_speed=0.0,
            mode_idx=self.mode_idx,
            ring_distance_prev=self._prev_ring_distance,
            ring_distance_cur=self._prev_ring_distance,
            initial_pos_real=np.array([start_x_real, start_y_real]),
            initial_theta=self.theta_state,
            reward_type=self.reward_type,
        )

        # invalidate distance-map cache
        self._rewardmap_cache.update({"mode_idx": None, "xlim": None, "ylim": None, "Z": None, "extent": None})

        return self.get_observation(), info

    # ============================== Utilities ==============================

    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _initialize_render(self, headless: bool = False):
        if self.fig is None:
            if not headless:
                plt.ion()
                self.initialized_ion = True
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

    # ---------- Distance field for background ----------
    def _compute_reward_field(self, xlim: Tuple[float, float], ylim: Tuple[float, float]):
        cx, cy = self.center_set[self.mode_idx]
        r0 = float(self.radius_set[self.mode_idx])

        nx = ny = max(8, self.reward_grid_res)
        xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
        ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)

        XX, YY = np.meshgrid(xs, ys, indexing='xy')
        d_center = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
        Z = np.abs(d_center - r0)  # distance-to-circle (>=0)
        extent = (xlim[0], xlim[1], ylim[0], ylim[1])
        return Z.astype(np.float32), extent

    def _maybe_draw_reward_map(self):
        if not self.visualize_reward or self.ax is None:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        cache = self._rewardmap_cache
        need_recompute = (
            cache["Z"] is None or
            cache["mode_idx"] != self.mode_idx or
            cache["xlim"] != tuple(xlim) or
            cache["ylim"] != tuple(ylim)
        )
        if need_recompute:
            Z, extent = self._compute_reward_field(tuple(xlim), tuple(ylim))
            cache.update({"Z": Z, "extent": extent, "mode_idx": self.mode_idx,
                          "xlim": tuple(xlim), "ylim": tuple(ylim)})

        # Smaller distance (closer to ring) looks better; we show raw distance.
        vmin = 0.0
        vmax = float(self.reward_contrast)  # env units
        self.ax.imshow(
            cache["Z"], extent=cache["extent"], origin="lower",
            interpolation="bilinear", cmap=self.reward_cmap,
            vmin=vmin, vmax=vmax, alpha=self.reward_alpha, zorder=0
        )

    # ============================== Rendering ==============================

    def render(self, headless: bool = False, camera_mode: str = 'follow', follow_zoom: float = 20.0) -> np.ndarray:
        """
        Returns an RGB image as np.ndarray (H, W, 3), dtype=uint8.
        """
        self._initialize_render(headless)
        assert self.ax is not None and self.fig is not None

        self.ax.clear()

        # set view first
        if camera_mode == 'fixed':
            if self.has_limits:
                x_min, x_max = self.limits[0]
                y_min, y_max = self.limits[1]
                self.ax.set_xlim(x_min * 1.1, x_max * 1.1)
                self.ax.set_ylim(y_min * 1.1, y_max * 1.1)
            else:
                lim = 40.0
                self.ax.set_xlim(-lim, lim)
                self.ax.set_ylim(-lim, lim)
        elif camera_mode == 'follow':
            agent_x, agent_y = self.state[0], self.state[1]
            self.ax.set_xlim(agent_x - follow_zoom, agent_x + follow_zoom)
            self.ax.set_ylim(agent_y - follow_zoom, agent_y + follow_zoom)
        else:
            raise ValueError(f"Unknown camera_mode: '{camera_mode}'. Use 'fixed' or 'follow'.")

        self.ax.set_aspect('equal', adjustable='box')

        # draw distance background first
        # self._maybe_draw_reward_map()

        # grid
        # self.ax.axhline(0, color='gray', linestyle='--', alpha=0.6, zorder=1)
        x_boundaries = np.array([-10.0, 0.0, 10.0]) / self.scale
        for x_val in x_boundaries:
            continue
            self.ax.axvline(x_val, color='gray', linestyle='--', alpha=0.6, zorder=1)

        # limits + shaded exterior
        if self.has_limits:
            x_min, x_max = self.limits[0]
            y_min, y_max = self.limits[1]
            self.ax.plot([x_min, x_max, x_max, x_min, x_min],
                         [y_min, y_min, y_max, y_max, y_min],
                         color='black', linestyle='-', linewidth=2, zorder=2)

            view_xlim = self.ax.get_xlim()
            view_ylim = self.ax.get_ylim()
            self.ax.axvspan(view_xlim[0], x_min, color='black', alpha=0.9, zorder=3)
            self.ax.axvspan(x_max, view_xlim[1], color='black', alpha=0.9, zorder=3)
            self.ax.axhspan(view_ylim[0], y_min, color='black', alpha=0.9, zorder=3)
            self.ax.axhspan(y_max, view_ylim[1], color='black', alpha=0.9, zorder=3)

        # quadrant labels (optional helper)
        text_style = {'fontsize': 14, 'ha': 'center', 'va': 'center', 'color': 'gray', 'alpha': 0.7}
        plot_xlim = self.ax.get_xlim()
        plot_ylim = self.ax.get_ylim()
        all_x_bounds = sorted(list(set([plot_xlim[0]] + x_boundaries.tolist() + [plot_xlim[1]])))

        x_centers: List[float] = []
        for i in range(len(all_x_bounds) - 1):
            if all_x_bounds[i+1] > plot_xlim[0] and all_x_bounds[i] < plot_xlim[1]:
                visible_region_start = max(all_x_bounds[i], plot_xlim[0])
                visible_region_end = min(all_x_bounds[i+1], plot_xlim[1])
                x_centers.append((visible_region_start + visible_region_end) / 2)

        y_regions = {1: (max(0, plot_ylim[0]) + plot_ylim[1]) / 2,
                     0: (plot_ylim[0] + min(0, plot_ylim[1])) / 2}

        start_x_quad_idx = 0
        if x_centers:
            first_center = x_centers[0]
            if first_center > x_boundaries[2]: start_x_quad_idx = 3
            elif first_center > x_boundaries[1]: start_x_quad_idx = 2
            elif first_center > x_boundaries[0]: start_x_quad_idx = 1

        for i, x_c in enumerate(x_centers):
            current_x_quad_idx = start_x_quad_idx + i
            for y_sign_idx, y_c in y_regions.items():
                combined_label = current_x_quad_idx * 2 + y_sign_idx
                # self.ax.text(x_c, y_c, str(combined_label), **text_style)

        # targets
        for i in range(self.num_modes):
            is_target = (i == self.mode_idx)
            continue
            circle_patch = Circle(
                self.center_set[i], self.radius_set[i],
                color='green' if is_target else 'gray',
                linestyle='-' if is_target else '--', fill=False,
                linewidth=2 if is_target else 1, alpha=1.0 if is_target else 0.5
            )
            self.ax.add_patch(circle_patch)

        # path + agent
        if self.path_history:
            path = np.array(self.path_history)
            self.ax.plot(path[:, 0], path[:, 1], color='blue', alpha=0.7, lw=1.5, label="Trajectory")

        x, y = self.state[0], self.state[1]
        self.ax.plot(x, y, 'ro', markersize=8, label="Agent")

        arrow_len = self.step_size * 2.0
        dx_arr = arrow_len * np.cos(self.theta_state)
        dy_arr = arrow_len * np.sin(self.theta_state)
        self.ax.arrow(x, y, dx_arr, dy_arr, head_width=arrow_len*0.2, head_length=arrow_len*0.3, fc='red', ec='red')

        # self.ax.set_title(f"Step: {self.t}, Mode: {self.mode_idx}", fontsize=10)
        # self.ax.set_xlabel("X-axis", fontsize=9)
        # self.ax.set_ylabel("Y-axis", fontsize=9)
        # self.ax.tick_params(axis='both', which='major', labelsize=8)

        # handles, labels = self.ax.get_legend_handles_labels()
        # if handles:
        #     self.ax.legend(handles, labels, loc="upper right", fontsize=8)
        plt.axis("off")   # <- hides axes
        
        self.fig.canvas.draw()
        image_from_canvas = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image_from_canvas.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if not headless:
            plt.draw()
            plt.pause(1.0 / self.metadata["render_fps"])

        return image

    def close(self):
        if self.fig is not None:
            if self.initialized_ion:
                plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None