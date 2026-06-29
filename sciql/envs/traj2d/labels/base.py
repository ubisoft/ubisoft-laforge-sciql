import numpy as np
from sciql.core.data import Episode
from sciql.core.label import EpisodeLabel
from tqdm import tqdm
from typing import List, Dict, Optional, Any # Ensure List is imported
from sciql.utils.labels import majority_window

def get_traj2d_episode_labels():
    labels = [PositionLabel(), MovementDirectionLabel(), TurnDirectionLabel(), RadiusCategoryLabel(), SpeedCategoryLabel(), CurvatureNoiseLabel()]
    return labels

# ================================================================= #
# 1. PositionLabel (Fully Refactored for Dynamic Binning)           #
# ================================================================= #
class PositionLabel(EpisodeLabel):
    def __init__(
        self,
        num_x_categories: int = 4,
        num_y_categories: int = 2,
        min_x_real: float = -30.0,
        max_x_real: float = 30.0,
        env_scale: float = 1.0,
        window_size: int = 1
    ):
        """
        Categorizes the agent's position into a 2D grid.

        Args:
            num_x_categories (int): Number of bins to create along the x-axis.
            num_y_categories (int): Number of bins for the y-axis (typically 2 for positive/negative).
            min_x_real (float): The minimum boundary for the x-axis range.
            max_x_real (float): The maximum boundary for the x-axis range.
            env_scale (float): The scaling factor of the environment.
        """
        super().__init__()
        self.name = 'position_label'
        if not min_x_real < max_x_real:
            raise ValueError("min_x_real must be less than max_x_real.")
        
        self.env_scale = env_scale
        self.num_x_categories = num_x_categories
        self.num_y_categories = num_y_categories
        
        # Automatically create bin edges for the x-axis.
        self.x_bin_edges = np.linspace(min_x_real, max_x_real, num_x_categories + 1)[1:-1]
        self.window_size = window_size
        self.num_labels = self.num_x_categories * self.num_y_categories
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        quadrant_labels = []
        for frame in episode:
            x_env_scaled, y_env_scaled = frame['observation']['full'][:2]
            x_real = x_env_scaled * self.env_scale
            y_real = y_env_scaled * self.env_scale

            x_quad_idx = int(np.digitize(x_real, self.x_bin_edges))
            y_quad_idx = 1 if y_real > 0 else 0
            
            combined_label = x_quad_idx * self.num_y_categories + y_quad_idx
            quadrant_labels.append(combined_label)
        
        return {self.name: majority_window(quadrant_labels, self.window_size)}

# ================================================================= #
# 2. MovementDirectionLabel (Verified Dynamic)                      #
# ================================================================= #
class MovementDirectionLabel(EpisodeLabel):
    def __init__(
        self, 
        num_direction_bins: int = 8, 
        env_scale: float = 1.0,
        window_size: int = 1
    ):
        super().__init__()
        self.name = 'movement_direction_label'
        if num_direction_bins <= 0:
            raise ValueError("num_direction_bins must be positive.")
        
        self.num_labels = num_direction_bins
        self.env_scale = env_scale
        self.bin_size_rad = (2 * np.pi) / self.num_labels
        self.window_size = window_size
        self.undetermined_label = self.num_labels
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels + 1))

    def _get_direction_bin(self, true_theta_rad: float) -> int:
        normalized_theta = (true_theta_rad + 2 * np.pi) % (2 * np.pi)
        bin_index = int(normalized_theta / self.bin_size_rad)
        return min(bin_index, self.num_labels - 1)

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        direction_labels = []
        for frame in episode:
            delta_position = frame['next_observation']['full'][:2] - frame['observation']['full'][:2]
            if np.linalg.norm(delta_position) * self.env_scale < 0.1:
                direction_labels.append(self.undetermined_label)
            else:
                angle_radians = np.arctan2(delta_position[1], delta_position[0])
                direction_labels.append(self._get_direction_bin(angle_radians))
        return {self.name: majority_window(direction_labels, self.window_size)}

# ================================================================= #
# 3. TurnDirectionLabel (with Centered Window)                      #
# ================================================================= #
class TurnDirectionLabel(EpisodeLabel):
    def __init__(
        self, 
        window_size: int = 10, 
        min_avg_rotation_rad_per_step: float = 0.1
    ):
        super().__init__()
        self.name = 'turn_direction_label'
        # Ensure window size is odd for a perfect center
        self.window_size = window_size + 1 if window_size % 2 == 0 else window_size
        self.min_avg_rotation = min_avg_rotation_rad_per_step
        self.num_labels = 3
        self.straight_label = 2
        self.promptable_labels = [0, 1]
        self.all_labels = [0, 1, 2]

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        if len(episode) < 2: return {self.name: [self.straight_label] * len(episode)}
        
        true_thetas = [f['infos']['true_theta'] for f in episode]
        unwrapped = np.unwrap(true_thetas)
        deltas = np.diff(unwrapped) # len is len(episode) - 1
        labels = []
        
        half_window = self.window_size // 2
        
        for t in range(len(episode)):
            # --- MODIFICATION: Centered window on the DELTAS ---
            # The delta at index `i` is for the step from `i` to `i+1`.
            # A window centered at `t` in the original timeline corresponds
            # to a window of deltas also centered around `t`.
            start = max(0, t - half_window)
            end = min(len(deltas), t + half_window + 1)
            
            label = self.straight_label
            relevant = deltas[start:end]
            
            if len(relevant) > 0:
                avg_vel = np.mean(relevant)
                if abs(avg_vel) >= self.min_avg_rotation:
                    label = 1 if avg_vel > 0 else 0
            labels.append(label)
        
        return {self.name: labels}

# ================================================================= #
# 4. RadiusCategoryLabel (with Centered Window & Theta Linearity)   #
# ================================================================= #
class RadiusCategoryLabel(EpisodeLabel):
    def __init__(
            self,
            num_radius_categories: int = 3,
            min_radius_real: float = 2.0,
            max_radius_real: float = 11.0,
            window_size: int = 50,
            straight_window_size: int = 10,
            env_scale: float = 1.0,
            min_avg_rotation_rad_per_step: float = 0.1,
    ):
        """
        Estimates radius by first checking for straightness using the consistency
        of the agent's orientation (theta).

        Args:
            num_radius_categories (int): Number of bins for the radius.
            min_radius_real (float): The minimum boundary for the radius range.
            max_radius_real (float): The maximum boundary for the radius range.
            window_size (int): The number of recent points to use for analysis.
            env_scale (float): The environment's scaling factor.
            straightness_theta_std_dev_threshold (float): The standard deviation of
                unwrapped theta values within the window. Below this, the path
                is considered a straight line.
        """
        super().__init__()
        self.name = 'radius_category_label'
        if not min_radius_real < max_radius_real:
            raise ValueError("min_radius_real must be less than max_radius_real.")
        
        # Ensure window size is odd for a perfect center
        assert window_size >= 10
        self.radius_window_size = window_size + 1 if window_size % 2 == 0 else window_size
        self.straight_window_size = straight_window_size + 1 if straight_window_size % 2 == 0 else straight_window_size
        self.env_scale = env_scale
        self.min_avg_rotation = min_avg_rotation_rad_per_step
        
        self.bin_edges = np.linspace(min_radius_real, max_radius_real, num_radius_categories + 1)[1:-1]
        
        self.straight_label = num_radius_categories
        self.num_labels = num_radius_categories + 1
        self.promptable_labels = list(range(num_radius_categories))
        self.all_labels = list(range(self.num_labels))

    @staticmethod
    def _fit_circle_least_squares(points: np.ndarray) -> float:
        if points.shape[0] < 3: return np.inf
        x, y = points[:, 0], points[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        b = x**2 + y**2
        try:
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            radius_squared = c[2] + c[0]**2 + c[1]**2
            return np.sqrt(radius_squared) if radius_squared >= 0 else np.inf
        except np.linalg.LinAlgError: return np.inf

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        if len(episode) < 3: return {self.name: [self.straight_label] * len(episode)}

        positions_scaled = np.array([frame['observation']['full'][:2] for frame in episode])
        positions_real = positions_scaled * self.env_scale
        true_thetas = [f['infos']['true_theta'] for f in episode]
        unwrapped_thetas = np.unwrap(true_thetas)
        deltas = np.diff(unwrapped_thetas)

        labels = []
        radius_half_window = self.radius_window_size // 2
        straight_half_window = self.straight_window_size // 2

        for t in range(len(episode)):

            start = max(0, t - radius_half_window)
            end = min(len(episode), t + radius_half_window + 1)
            position_window = positions_real[start:end]

            start = max(0, t - straight_half_window)
            end = min(len(episode), t + straight_half_window + 1)
            relevant = deltas[start:end]
            
            if len(relevant) == 0:
                label = self.straight_label
            elif np.abs(np.mean(relevant)) < self.min_avg_rotation:
                label = self.straight_label
            else:
                # Only if the path is curved, fit a circle to the positions.
                radius = self._fit_circle_least_squares(position_window)
                if np.isinf(radius):
                    label = self.straight_label
                else:
                    label = int(np.digitize(radius, self.bin_edges))
                
            labels.append(label)
            
        return {self.name: labels}

# ================================================================= #
# 5. SpeedCategoryLabel (Verified Dynamic)                          #
# ================================================================= #
class SpeedCategoryLabel(EpisodeLabel):
    def __init__(
        self, 
        min_speed_real: float = 0.5, 
        max_speed_real: float = 3.0, 
        num_categories: int = 3,
        window_size: int = 1
    ):
        super().__init__()
        self.name = 'speed_category_label'
        if not min_speed_real < max_speed_real:
            raise ValueError("min_speed_real must be less than max_speed_real.")
        
        self.num_labels = num_categories
        self.bin_edges = np.linspace(min_speed_real, max_speed_real, num_categories + 1)[1:-1]
        self.window_size = window_size
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        speed_labels = []
        for frame in episode:
            if 'true_speed' not in frame['infos']:
                raise ValueError("Frame infos must contain 'true_speed' for this label.")
            category = np.digitize(frame['infos']['true_speed'], self.bin_edges)
            speed_labels.append(int(category))
        return {self.name: majority_window(speed_labels, self.window_size)}

# ================================================================= #
# 6. CurvatureNoiseLabel (with Centered Window)                     #
# ================================================================= #
class CurvatureNoiseLabel(EpisodeLabel):
    def __init__(
        self, 
        min_noise_std_dev: float = 0.0, 
        max_noise_std_dev: float = 0.8, 
        num_categories: int = 3, 
        window_size: int = 50
    ):
        super().__init__()
        self.name = 'curvature_noise_label'
        if not min_noise_std_dev < max_noise_std_dev:
            raise ValueError("min_noise_std_dev must be less than max_noise_std_dev.")
        
        # Ensure window size is odd for a perfect center
        self.window_size = window_size + 1 if window_size % 2 == 0 else window_size
        self.num_labels = num_categories
        self.bin_edges = np.linspace(min_noise_std_dev, max_noise_std_dev, num_categories + 1)[1:-1]
        
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        if len(episode) < 4: return {self.name: [0] * len(episode)}
        
        thetas = [f['infos']['true_theta'] for f in episode]
        unwrapped = np.unwrap(thetas)
        deltas = np.diff(unwrapped)
        delta2s = np.diff(deltas) # len is len(episode) - 2
        labels = []
        
        half_window = self.window_size // 2

        for t in range(len(episode)):
            # --- MODIFICATION: Centered window on the DELTA2s ---
            # A window centered at `t` in the original timeline corresponds
            # to a window of delta2s also centered around `t`.
            start = max(0, t - half_window)
            end = min(len(delta2s), t + half_window + 1)
            
            relevant = delta2s[start:end]
            
            if len(relevant) < 2:
                noise = 0.0
            else:
                noise = np.std(relevant)
                
            labels.append(int(np.digitize(noise, self.bin_edges)))
            
        return {self.name: labels}

# ================================================================= #
# 7. DisplacementLabel                                             #
# ================================================================= #
class DisplacementLabel(EpisodeLabel):
    def __init__(
        self,
        min_displacement_real: float = 0.0,
        max_displacement_real: float = 50.0,
        num_categories: int = 3,
        env_scale: float = 1.0,
        window_size: int = 1
    ):
        """
        Categorizes the agent's current displacement from its starting state (s0).

        Args:
            min_displacement_real (float): The minimum boundary for the displacement range.
            max_displacement_real (float): The maximum boundary for the displacement range.
            num_categories (int): The number of bins (e.g., 3 for "Near", "Mid-range", "Far").
            env_scale (float): The scaling factor of the environment to convert units.
        """
        super().__init__()
        self.name = 'displacement_label'
        if not min_displacement_real < max_displacement_real:
            raise ValueError("min_displacement_real must be less than max_displacement_real.")

        self.env_scale = env_scale
        self.num_labels = num_categories
        
        # Create bin edges for the displacement categories.
        self.bin_edges = np.linspace(min_displacement_real, max_displacement_real, num_categories + 1)[1:-1]
        self.window_size = window_size
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        displacement_labels = []
        

        s0_coords_scaled = episode[0]['observation']['full'][:2]
        for frame in episode:
            obs = frame['observation']['full']
            
            # The observation format is [s0, s_t, s_{t-1}, ...].
            # s0's coordinates are the first two elements.
            # s_t's coordinates are the first two elements of the history stack.
            st_coords_scaled = obs[:2] # Current state is at the start of the history part

            # Convert from environment units to real-world units for distance calculation.
            s0_coords_real = s0_coords_scaled * self.env_scale
            st_coords_real = st_coords_scaled * self.env_scale
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(st_coords_real - s0_coords_real)
            
            # Categorize the distance
            category = int(np.digitize(distance, self.bin_edges))
            displacement_labels.append(category)

        return {self.name: majority_window(displacement_labels, self.window_size)}

def plot_trajectories_with_labels(
    episodes: List[Episode],
    labeler_instance: EpisodeLabel,
    title_suffix: str = "",
    arrow_every_n_steps: int = 5, # Draw an arrow every N steps
    arrow_length_factor: float = 0.05 # Factor to scale arrow length (relative to plot scale)
):
    """
    Generates labels for multiple episodes using the given labeler and plots all
    trajectories on a single figure, with points colored by their labels,
    and arrows indicating movement direction.
    """
    print(f"\n--- Plotting {len(episodes)} episodes with Labeler: {labeler_instance.name} ---")

    import matplotlib.pyplot as plt
    
    all_episode_plot_data = [] # Stores {'obs_xy':, 'thetas':, 'labels':, 'id':}
    all_unique_labels = set()

    for ep_idx, episode in enumerate(episodes):
        if not episode or len(episode) == 0:
            print(f"Episode {ep_idx} is empty, skipping.")
            continue

        # Get labels
        try:
            labels_dict = labeler_instance(episode)
            if labeler_instance.name not in labels_dict:
                print(f"Error (Ep {ep_idx}): Labeler {labeler_instance.name} did not return its own name as a key.")
                continue
            labels = labels_dict[labeler_instance.name]
        except Exception as e:
            print(f"Error (Ep {ep_idx}) generating labels with {labeler_instance.name}: {e}")
            continue

        if not isinstance(labels, list) or len(labels) != len(episode):
            print(f"Error (Ep {ep_idx}): Labels are not a list or num labels ({len(labels)}) != ep length ({len(episode)}).")
            continue
        
        all_unique_labels.update(labels)

        # Extract observations (x, y) and true_thetas
        try:
            obs_xy_list = []
            true_thetas_list = []
            for frame_idx, frame in enumerate(episode):
                # Access observation data (x, y, theta_obs)
                # Based on your `obs = np.stack([frame['observation']['full'] ...`
                # I'm assuming frame is dict-like or Frame class handles this.
                # For my dummy Frame, it's frame.observation_data
                current_obs_data = frame['observation']['full']
                obs_xy_list.append(current_obs_data[:2]) # x, y

                if 'true_theta' not in frame['infos']:
                    raise ValueError(f"Frame {frame_idx} in episode {ep_idx} missing 'true_theta' in infos.")
                true_thetas_list.append(frame['infos']['true_theta'])
            
            if not obs_xy_list:
                print(f"Error (Ep {ep_idx}): No observations extracted.")
                continue
            
            all_episode_plot_data.append({
                'obs_xy': np.array(obs_xy_list),
                'thetas': np.array(true_thetas_list),
                'labels': labels,
                'id': ep_idx
            })

        except (KeyError, TypeError, AttributeError, ValueError) as e:
            print(f"Error (Ep {ep_idx}) processing frame data: {e}.")
            continue

    if not all_episode_plot_data:
        print("No valid episode data to plot.")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sorted_unique_labels = sorted(list(all_unique_labels))
    num_distinct_categories = len(sorted_unique_labels)

    cmap = None
    label_to_color_idx = {}
    if num_distinct_categories > 0:
        if num_distinct_categories <= 10: cmap = plt.get_cmap('tab10', num_distinct_categories)
        elif num_distinct_categories <= 20: cmap = plt.get_cmap('tab20', num_distinct_categories)
        else: cmap = plt.get_cmap('viridis', num_distinct_categories)
        label_to_color_idx = {label_val: i for i, label_val in enumerate(sorted_unique_labels)}

    # Determine overall plot scale for arrow length
    all_x = np.concatenate([ep_data['obs_xy'][:, 0] for ep_data in all_episode_plot_data if len(ep_data['obs_xy']) > 0])
    all_y = np.concatenate([ep_data['obs_xy'][:, 1] for ep_data in all_episode_plot_data if len(ep_data['obs_xy']) > 0])
    if len(all_x) == 0 or len(all_y) == 0: # Should not happen if all_episode_plot_data is not empty
        plot_x_range = 1.0
    else:
        plot_x_range = np.ptp(all_x) if len(all_x) > 1 else 1.0 # Peak-to-peak
    base_arrow_length = plot_x_range * arrow_length_factor


    for ep_data in all_episode_plot_data:
        obs_xy = ep_data['obs_xy']
        thetas = ep_data['thetas']
        labels = ep_data['labels']
        
        if len(obs_xy) == 0: continue

        # Get colors for scatter points
        point_colors = None
        if cmap and label_to_color_idx:
            color_indices = [label_to_color_idx.get(l, 0) for l in labels] # Default to 0 if label somehow missing
            point_colors = cmap(np.array(color_indices) / (num_distinct_categories -1 if num_distinct_categories > 1 else 1.0) )


        # Plot trajectory line (faintly)
        ax.plot(obs_xy[:, 0], obs_xy[:, 1], color='gray', alpha=0.4, linewidth=0.7, zorder=0)

        # Scatter plot for points, colored by label
        if point_colors is not None:
            ax.scatter(obs_xy[:, 0], obs_xy[:, 1], c=point_colors, 
                       s=20, alpha=0.5, zorder=1)
        else: # If no labels or cmap not set
            ax.scatter(obs_xy[:, 0], obs_xy[:, 1], color='blue', s=20, alpha=0.5, zorder=1)

        # Add arrows using plt.quiver for efficiency
        arrow_x_pos = []
        arrow_y_pos = []
        arrow_dx = []
        arrow_dy = []
        arrow_colors = []

        for i in range(0, len(obs_xy), arrow_every_n_steps):
            if i < len(thetas): # Ensure theta is available for this point
                arrow_x_pos.append(obs_xy[i, 0])
                arrow_y_pos.append(obs_xy[i, 1])
                arrow_dx.append(base_arrow_length * np.cos(thetas[i]))
                arrow_dy.append(base_arrow_length * np.sin(thetas[i]))
                if point_colors is not None:
                    arrow_colors.append(point_colors[i])
                else:
                    arrow_colors.append('black') # Default arrow color

        if arrow_x_pos: # If any arrows to plot
            ax.quiver(arrow_x_pos, arrow_y_pos, arrow_dx, arrow_dy,
                      color=arrow_colors, scale_units='xy', angles='xy', scale=1,
                      width=0.003, headwidth=3, headlength=4, zorder=2, alpha=0.5)

    # Create a legend for the labels
    if cmap and label_to_color_idx and num_distinct_categories > 0:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Label {ul}',
                                      markerfacecolor=cmap(label_to_color_idx[ul] / (num_distinct_categories -1 if num_distinct_categories > 1 else 1.0)), 
                                      markersize=8)
                           for ul in sorted_unique_labels]
        ax.legend(handles=legend_elements, title="Labels", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax.set_title(f"Trajectories ({len(all_episode_plot_data)} eps) by {labeler_instance.name}{title_suffix}", fontsize=14)
    ax.set_xlabel("X coordinate (scaled)", fontsize=12)
    ax.set_ylabel("Y coordinate (scaled)", fontsize=12)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axvline(0, color='black', linestyle=':', linewidth=0.7, alpha=0.5)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.show()