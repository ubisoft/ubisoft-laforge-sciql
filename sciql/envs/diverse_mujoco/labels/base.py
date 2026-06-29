import numpy as np
from typing import Dict, List, Optional, Iterable, Any
from sciql.core.label import EpisodeLabel
from sciql.core.data import Episode

class EmptyLabel(EpisodeLabel):
    def __init__(self):

        super().__init__()
        self.name = 'empty_label'
        self.num_labels = 10
        self.promptable_labels = list(range(self.num_labels))
        self.all_labels = list(range(self.num_labels))

    def __call__(self, episode: Episode) -> Dict[str, List[int]]:
        return {self.name: np.random.randint(low=0, high=self.num_labels, size=len(episode)).tolist()}
    
def get_episode_labels(env_name: str):
    if 'HalfCheetah' in env_name:
        from .halfcheetah import SpeedLabel, DirectionLabel, AngleLabel, TorsoHeightLabel, FrequencyLabel, BackfootHeightLabel, FrontfootHeightLabel
        labels = [SpeedLabel(), AngleLabel(), TorsoHeightLabel(), BackfootHeightLabel(), FrontfootHeightLabel(), EmptyLabel()]
    else:
        raise ValueError('Unknown env_name.')
    return labels