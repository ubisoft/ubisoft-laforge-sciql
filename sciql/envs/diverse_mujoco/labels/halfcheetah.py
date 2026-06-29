import numpy as np
from typing import Dict, List, Optional, Iterable, Any
from sciql.core.label import EpisodeLabel
from sciql.core.data import Episode
from sciql.utils.labels import majority_window

class SpeedLabel(EpisodeLabel):
    def __init__(
        self,
        min_speed: float = 0.1,
        max_speed: float = 10.0,
        num_categories: int = 3,
        window_size: int = 1
    ):
        """
        Categorizes the agent's forward speed into dynamic bins.

        Args:
            min_speed (float): The minimum speed boundary for categorization.
            max_speed (float): The maximum speed boundary for categorization.
            num_categories (int): The number of bins to create between min_speed and max_speed.
            no_movement_threshold (float): The threshold below which speed is considered stationary.
        """
        super().__init__()
        self.name = 'speed_label'
        if not min_speed < max_speed:
            raise ValueError("min_speed must be less than max_speed.")

        # Create bin edges for categorization. np.digitize uses these as right-inclusive boundaries.
        self.speed_bins = np.linspace(min_speed, max_speed, num_categories + 1)[1:-1]
        print(f'Speed bins: {self.speed_bins}')
        
        # Total labels = num_categories + 1 (for the stationary category)
        self.window_size = window_size
        self.num_labels = num_categories
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        speed_labels = []
        for frame in episode:
            velocity = np.abs(frame['infos']['x_velocity'])
            # Digitize and add 1 to offset for the stationary category
            label = int(np.digitize(velocity, self.speed_bins))
            speed_labels.append(label)
        
        return {self.name: majority_window(speed_labels, self.window_size)}

class DirectionLabel(EpisodeLabel):
    def __init__(
        self, 
        stationary_threshold: float = 0.1, 
        window_size: int = 1
    ):
        """
        Categorizes the agent's direction of movement.

        Args:
            stationary_threshold (float): The speed threshold to be considered stationary.
        """
        super().__init__()
        self.name = 'direction_label'
        self.stationary_threshold = stationary_threshold
        self.window_size = window_size
        self.num_labels = 3  # 0: Stationary, 1: Forward, 2: Backward
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: 'Episode') -> Dict[str, List[int]]:
        direction_labels = []
        for frame in episode:
            velocity = frame['infos']['x_velocity']
            if abs(velocity) < self.stationary_threshold:
                direction_labels.append(0)  # Stationary
            elif velocity > 0:
                direction_labels.append(1)  # Forward
            else:
                direction_labels.append(2)  # Backward
        
        return {self.name: majority_window(direction_labels, self.window_size)}

class AngleLabel(EpisodeLabel):
    def __init__(
        self,
        min_angle: float = -0.3,
        max_angle: float = 0.3,
        num_categories: int = 3,
        window_size: int = 1
    ):
        """
        Categorizes the agent's torso angle into dynamic bins.

        Args:
            min_angle (float): The minimum angle boundary (e.g., for 'Upright').
            max_angle (float): The maximum angle boundary (e.g., for 'Crouching').
            num_categories (int): The number of bins to create for the angle range.
        """
        super().__init__()
        self.name = 'angle_label'
        if not min_angle < max_angle:
            raise ValueError("min_angle must be less than max_angle.")

        self.angle_bins = np.linspace(min_angle, max_angle, num_categories + 1)[1:-1]
        print(f'Angle bins: {self.angle_bins}')
        self.window_size = window_size
        self.num_labels = num_categories
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        angle_labels = []
        for frame in episode:
            torso_angle = frame['infos']['torso_angle']
            label = int(np.digitize(torso_angle, self.angle_bins))
            angle_labels.append(label)

        return {self.name: majority_window(angle_labels, self.window_size)}
    
class FrequencyLabel(EpisodeLabel):
    def __init__(
        self,
        min_freq: float = 1.0,
        max_freq: float = 5.0,
        num_categories: int = 3,
        window_size: int = 1
    ):
        """
        Categorizes the agent's stepping frequency into dynamic bins.

        Args:
            min_freq (float): The minimum frequency for categorization.
            max_freq (float): The maximum frequency for categorization.
            num_categories (int): The number of bins to create for the frequency range.
        """
        super().__init__()
        self.name = 'frequency_label'
        if not min_freq < max_freq:
            raise ValueError("min_freq must be less than max_freq.")

        self.frequency_bins = np.linspace(min_freq, max_freq, num_categories + 1)[1:-1]
        
        # Total labels = num_categories + 1 (for the 'no step' category)
        self.window_size = window_size
        self.num_labels = num_categories + 1
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        frequency_labels = []
        for frame in episode:
            step_freq = frame['infos']['step_freq']
            if step_freq == 0:
                frequency_labels.append(0)  # Category 0: No step
            else:
                # Digitize and add 1 to offset for the 'no step' category
                label = int(np.digitize(step_freq, self.frequency_bins)) + 1
                frequency_labels.append(label)
        
        return {self.name: majority_window(frequency_labels, self.window_size)}

class TorsoHeightLabel(EpisodeLabel):
    def __init__(
        self,
        min_height: float = 0.4,
        max_height: float = 0.8,
        num_categories: int = 3,
        window_size: int = 1
    ):
        """
        Categorizes the agent's torso height into dynamic bins.

        Args:
            min_height (float): The minimum height boundary (e.g., for 'Crawling').
            max_height (float): The maximum height boundary (e.g., for 'Running').
            num_categories (int): The number of bins to create for the height range.
        """
        super().__init__()
        self.name = 'torso_height_label'
        if not min_height < max_height:
            raise ValueError("min_height must be less than max_height.")

        self.height_bins = np.linspace(min_height, max_height, num_categories + 1)[1:-1]
        print(f'TorsoHeight bins: {self.height_bins}')
        self.window_size = window_size
        self.num_labels = num_categories
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        height_labels = []
        for frame in episode:
            torso_height = frame['infos']['torso_z']
            label = int(np.digitize(torso_height, self.height_bins))
            height_labels.append(label)

        return {self.name: majority_window(height_labels, self.window_size)}
    
class BackfootHeightLabel(EpisodeLabel):
    def __init__(
        self,
        min_height: float = 0.0,
        max_height: float = 0.3,
        num_categories: int = 4,
        window_size: int = 1
    ):
        """
        Categorizes the height of the agent's back foot into dynamic bins.

        Args:
            min_height (float): The minimum height boundary (e.g., ground contact).
            max_height (float): The maximum height boundary (e.g., peak of stride).
            num_categories (int): The number of bins to create for the height range.
        """
        super().__init__()
        self.name = 'backfoot_height_label'
        if not min_height < max_height:
            raise ValueError("min_height must be less than max_height.")
        
        self.bfoot_height_bins = np.linspace(min_height, max_height, num_categories + 1)[1:-1]
        print(f'BackfootHeight bins: {self.bfoot_height_bins}')
        self.window_size = window_size
        self.num_labels = num_categories
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        bfoot_labels = []
        for frame in episode:
            bfoot_z = frame['infos']['bfoot_z']
            label = int(np.digitize(bfoot_z, self.bfoot_height_bins))
            bfoot_labels.append(label)
        
        return {self.name: majority_window(bfoot_labels, self.window_size)}

class FrontfootHeightLabel(EpisodeLabel):
    def __init__(
        self,
        min_height: float = 0.0,
        max_height: float = 0.3,
        num_categories: int = 4,
        window_size: int = 1
    ):
        """
        Categorizes the height of the agent's front foot into dynamic bins.

        Args:
            min_height (float): The minimum height boundary (e.g., ground contact).
            max_height (float): The maximum height boundary (e.g., peak of stride).
            num_categories (int): The number of bins to create for the height range.
        """
        super().__init__()
        self.name = 'frontfoot_height_label'
        if not min_height < max_height:
            raise ValueError("min_height must be less than max_height.")

        self.ffoot_height_bins = np.linspace(min_height, max_height, num_categories + 1)[1:-1]
        print(f'FrontfootHeight bins: {self.ffoot_height_bins}')
        self.window_size = window_size
        self.num_labels = num_categories
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        ffoot_labels = []
        for frame in episode:
            ffoot_z = frame['infos']['ffoot_z']
            label = int(np.digitize(ffoot_z, self.ffoot_height_bins))
            ffoot_labels.append(label)
        
        return {self.name: majority_window(ffoot_labels, self.window_size)}
    



