from abc import ABC, abstractmethod
from typing import Dict
from sciql.core.data import Episode

class EpisodeStatistic(ABC):
    """
    An `EpisodeStatistic` is a function that takes in an episode an computes a statistic from it,
    like the discounted sum of rewards.
    """
    def __init__(self):
        self.name = 'episode_statistic'

    @abstractmethod
    def __call__(self, episode: Episode) -> Dict:
        raise NotImplementedError