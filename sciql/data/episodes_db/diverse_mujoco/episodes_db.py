import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from sciql.core.data import BaseEpisode, EpisodesDB
from typing import Any, Iterator, List, Tuple, Union
import os
from sciql.utils.paths import DATASETS_PATH
from sciql.data.episodes_db.diverse_mujoco.datasets import load_diverse_mujoco_dataset

class DiverseMujoco_EpisodesDB(EpisodesDB):

    """
    A database of episodes from OGBench datasets.

    Args:
        env (str): The environment name.
    """

    def __init__(self, dataset_dir: str, archetypes: list[str] = None) -> None:

        # Get dataset
        dataset_dir = os.path.join(DATASETS_PATH, dataset_dir)

        if archetypes is None:
            if 'fix' in dataset_dir:
                archetypes = ['fix']
            elif 'vary' in dataset_dir:
                archetypes = ['vary']
            elif 'stitch' in dataset_dir:
                archetypes = ['stitch']
            else:
                archetypes = ['Angle_Crouching', 'Angle_Flat', 'Angle_Upright', 'Height_Crawling', 'Height_Normal', 'Height_Running', 'Speed_Fast', 'Speed_Medium', 'Speed_Slow']

        dataset = load_diverse_mujoco_dataset(dataset_dir=dataset_dir, archetypes=archetypes)
        
        # Get data
        self._observation:np.ndarray = dataset["observations"]
        self._action:np.ndarray = dataset["actions"]
        self._next_observation:np.ndarray = dataset["next_observations"]
        self._reward:np.ndarray = dataset["rewards"]
        self._terminated:np.ndarray = dataset["terminated"]
        self._truncated:np.ndarray = dataset["truncated"]
        self._infos:dict[np.ndarray] = {k: dataset[k] for k in dataset.keys() if 'infos' in k}
        self._done:np.ndarray = np.logical_or(self._terminated, self._truncated).astype(np.float32)
        
        # Get episodes infos
        end_idxs = np.where(self._done)[0]
        start_idxs = np.concatenate([[0],end_idxs[:-1] + 1])
        self._ids = [f"episode_{i+1}" for i in range(len(end_idxs))]
        self._start_end = {episode_id: (start_idxs[i],end_idxs[i]) for i,episode_id in enumerate(self._ids)}

    def __contains__(self, episode_id: str) -> bool:
        """
        Returns True if the episode_id is in the database.

        Args:
            episode_id (str): The episode ID.
        
        Returns:
            (bool) True if the episode_id is in the database.
        """
        return episode_id in self.get_ids()

    def __getitem__(self, episode_id: Union[int, str]) -> Union[BaseEpisode,dict[str,np.ndarray]]:

        """
        Returns the episode with the given episode_id.

        Args:
            episode_id (str): The episode ID.

        Returns:
            (Episode) The episode with the given episode_id.
        """

        # Get indxs
        _episode_id = episode_id if isinstance(episode_id,str) else self.get_ids()[episode_id]
        start_idx = self._start_end[_episode_id][0]
        end_idx= self._start_end[_episode_id][1]
        length = end_idx + 1 - start_idx
        
        # Return episode
        episode_dict = {
            "observation/full": self._observation[start_idx:end_idx+1],
            "action/full": self._action[start_idx:end_idx+1],
            "next_observation/full":self._next_observation[start_idx:end_idx+1],
            "reward": self._reward[start_idx:end_idx+1],
            "terminated": self._terminated[start_idx:end_idx+1],
            "truncated": self._truncated[start_idx:end_idx+1],
        }
        episode_dict.update({k: v[start_idx:end_idx+1] for k, v in self._infos.items()})

        return BaseEpisode.from_dict(episode_dict, _episode_id)

    def __iter__(self) -> Iterator[BaseEpisode]:
        """
        Returns an iterator over the episodes in the database.

        Returns:
            (Iterator[Episode]): An iterator over the episodes in the database.
        """
        ids = self.get_ids()
        for episode_id in ids:
            yield self[episode_id]

    def __len__(self) -> int:
        """
        Returns the number of episode_ids in the database.

        Returns:
            (int): The number of episode_ids in the database.
        """
        return len(self.get_ids())

    def __repr__(self) -> str:
        """
        Returns a string representation of the database.

        Returns:
            (str): A string representation of the database.
        """
        return f"OGBenchEpisodesDB(env={self._env_name}) with {len(self)} episodes."

    def add_episode(self, episode: BaseEpisode) -> None:
        """
        Impossible to add an episode to the database. Raises an error.

        Args:
            episode (Episode): The episode to add.
        """
        raise NotImplementedError("Impossible to add an episode to the database.")

    def delete_episode(self, episode_id: str) -> None:
        """
        Deletes an episode from the database.

        Args:
            episode_id (str): The episode ID.
        
        Returns:
            None
        """
        assert episode_id in self.get_ids()
        self._ids.remove(episode_id)

    def pop(self, episode_id: str) -> BaseEpisode:
        """
        Deletes an episode from the database and returns it.

        Args:
            episode_id (str): The episode ID.

        Returns:
            (Episode): The deleted episode.
        """
        assert episode_id in self.get_ids()
        episode = self[episode_id]
        self.delete_episode(episode_id)
        return episode

    def get_ids(self) -> List[str]:
        """
        Returns the episode_ids in the database.

        Returns:
            (List[str]): The episode_ids in the database.
        """
        return self._ids
        
