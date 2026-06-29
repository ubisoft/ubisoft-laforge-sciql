from abc import ABC, abstractmethod
from typing import Dict
from sciql.core.data import Episode

class EpisodeLabel(ABC):
    """
    An `EpisodeLabel` is a function that takes in an sequence
    in the form of an episode or a Dict and returns a dict of 
    transitions labels.
    """
    def __init__(self):
        self.name = 'episode_labeler'
        self.promptable_labels = {}
        self.all_labels = {}
    
    def get_promptable_labels_number(self):
        """
        Give the number of promptable labels.
        """
        return len(self.promptable_labels)
    
    def get_promptable_labels(self):
        """
        Get the idxs of all promptable labels.
        """
        return self.promptable_labels
    
    def is_promptable_idx(self, idx: int):
        """
        Checks if label id is promptable.
        """
        return idx in self.get_promptable_labels()

    @abstractmethod
    def __call__(self, episode: Episode) -> Dict:
        raise NotImplementedError
    
