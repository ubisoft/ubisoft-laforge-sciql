import numpy as np
import random as rd
import uuid

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Any, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union

Observation = Dict[str, np.ndarray]
Action = Dict[str, np.ndarray]

def flatten_dict(dictionary: Dict[str, Any], parent_key: str = '', separator: str = '/') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        dictionary (Dict[str, Any]): The dictionary to flatten.
        parent_key (str): The parent key. Defaults to ''.
        separator (str): The separator for the keys. Defaults to '/'.

    Returns:
        flattened_dictionary (Dict[str, Any]): The flattened dictionary.
    """
    # stack dictionaries and their parent keys (to avoid using recursion)
    stack: List[Tuple[Dict[str, Any], str]] = [(dictionary, parent_key)]
    flattened_dictionary: Dict[str, Any] = {}
    while stack:
        current_dictionary, current_key = stack.pop()
        for key, value in current_dictionary.items():
            new_key = f"{current_key}{separator}{key}" if current_key else key
            if isinstance(value, dict): stack.append((value, new_key))
            else: flattened_dictionary[new_key] = value
    return flattened_dictionary

def unflatten_dict(dictionary: Dict[str, Any], separator: str = '/') -> Dict[str, Any]:
    """
    Unflatten a dictionary into a nested one.

    Args:
        dictionary (Dict[str, Any]): The dictionary to unflatten.
        separator (str): The separator for the keys. Defaults to '/'.

    Returns:
        unflattened_dictionary (Dict[str, Any]): The unflattened dictionary.
    """
    unflattened_dictionary = {}
    for key, value in dictionary.items():
        keys = key.split(separator)
        current_dict = unflattened_dictionary
        for other_key in keys[:-1]:
            # use setdefault to avoid overwriting existing dictionaries
            current_dict = current_dict.setdefault(other_key, {})
        current_dict[keys[-1]] = value
    return unflattened_dictionary

class ImmutableMapping(MutableMapping):

    """
    An abstract class that defines read-only and immutable mappings.
    """
    
    __slots__ = []
    
    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} object is immutable.")

    def __delitem__(self, key: Any) -> NoReturn:
        raise TypeError(f"{type(self).__name__} object is immutable.")

class Frame(MutableMapping):

    """
    A data structure that describes the state of an `Episode` at a given timestep.
    It is similar to a dictionary but does not provide any write access.

    The dimension of the arrays in the frame is `(batch_size, array_size)`.

    Args:
        observation (Dict[str, np.ndarray]): The observation/state at the current time step.
        action (Dict[str, np.ndarray]): The action taken by the agent.
        reward (np.ndarray): The reward received after taking the action.
        terminated (np.ndarray): Indicates if the next state is a terminal state.
        truncated (np.ndarray): Indicates if the next state is truncated.
    """

    __slots__ = ["_data", "_flattened_cache"]

    def __init__(
        self,
        observation: Dict[str, np.ndarray] = None, 
        action: Dict[str, np.ndarray] = None,
        reward: np.ndarray = None,
        next_observation: Dict[str, np.ndarray] = None,
        terminated: np.ndarray = None,
        truncated: np.ndarray = None,
        infos: Dict = None) -> None:

        self._data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "terminated": terminated,
            "truncated": truncated,
            "infos": infos
        }

        self._flattened_cache: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = None

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        return self._data[key]
    
    def __setitem__(self, key: Any, value: Any) -> NoReturn:
        self._data[key] = value

    def __delitem__(self, key: Any) -> NoReturn:
        if key in self._data:
            del self._data[key]
            self._flattened_cache = None  # Clear cache if data is changed
        else:
            raise KeyError(f"Key '{key}' not found in Frame.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self) -> str:
        return f"Frame({', '.join([f'{key}={value}' for key, value in self._data.items()])})"

    def items(self) -> Generator[tuple[str, Any], Any, None]:
        for key in self:
            yield key, self._data[key]

    def to_dict(self) -> Dict[str, np.ndarray]:
        if self._flattened_cache is None:
            self._flattened_cache = flatten_dict(self._data)
        return flatten_dict(self._data)
    
    @staticmethod
    def from_dict(data: Dict[str, np.ndarray]) -> "Frame":
        return Frame(**unflatten_dict(data))

class Episode(ABC):

    """
    A class representing an episode, which is a sequence of frames in a reinforcement learning environment.
    It is expected to have a form similar to:
    {
        'observation': Dict[str, np.ndarray[T, ...]], T the episode length         -> [o_0, o_1, o_2, ..., o_T-1] (the sequence of observations, possibly of various types)
        'action': Dict[str, np.ndarray[T, ...]], T the episode length              -> [a_0, a_1, a_2, ..., a_T-1] (the sequence of actions, possibly of various types)
        'reward': np.ndarray[T, ...], T the episode length              -> [r_0, r_1, r_2, ..., r_T-1] (the sequence of rewards)
        'next_observation': Dict[str, np.ndarray[T, ...]], T the episode length         -> [o_1, o_1, o_2, ..., o_T] (the sequence of next_observations, possibly of various types)
        'terminated': np.ndarray[T, ...], T the episode length            -> [0, 0, ..., 0 or 1]         (if the state of next_observations is a real terminal state)
        'truncated': np.ndarray[T, ...], T the episode length             -> [0, 0, ..., 0 or 1]       (if the state of next_observations as been timed out)
    }
    """

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        """
        Checks if the key is present in the episode.
        """
        raise NotADirectoryError

    @abstractmethod
    def __getitem__(self, t: int) -> Frame:
        """
        Loads the frame of timestep t from the episode.
        """
        raise NotImplementedError

    def __setitem__(self, t: int, frame: Frame) -> None:
        """
        If possible, sets the provided frame as the new frame of timestep t of the episode. Not abstract
        since not all episodes are adapted to this feature.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        """
        Returns an iterator on the episode frames.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the episodes as the number of frames.
        """
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns the representation of the episode as a string.
        """
        raise NotImplementedError

    def get_id(self) -> str:
        """
        Returns the id of the episode.
        """
        return self._episode_id

    def add_frame(self, frame: Frame) -> None:
        """
        Adds a frame at the end of the episode. Not adapted to all episodes.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns the episodes data as a flattened dictionnary. Not adapted to all episodes.
        """
        raise NotImplementedError
    
    def from_dict(data: Dict[str, np.ndarray], episode_id: str = None) -> "Episode":
        """
        Returns the dict data as an episode.  Not adapted to all episodes.
        """
        raise NotImplementedError

class BaseEpisode(Episode):

    __slots__ = ["_episode_id", "_empty", "_data"]
    
    def __init__(self, episode_id: str = None) -> None:

        self._episode_id: str = episode_id if episode_id else str(uuid.uuid4())
        self._empty: bool = True
        self._data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]] = {
            "observation": None,
            "action": None,
            "reward": None,
            "next_observation": None,
            "terminated": None,
            "truncated": None,
            "infos": None
        }
        self._metadata: Dict = {}
    
    def add_metadata(self, key: str, value: Any):
        if key in self._metadata:
            print(f'Overwriting key: {key}')
        self._metadata[key] = value
    
    def get_metadata(self):
        return self._metadata

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, t: int) -> Frame:
        if self._empty:
            raise IndexError("Episode is empty.")
        frame_dict = {}
        for key, value in self._data.items():
            if isinstance(value, np.ndarray):
                frame_dict[key] = value[t]
            elif isinstance(value, dict):
                frame_dict[key] = {sub_key: sub_value[t] for sub_key, sub_value in value.items()}
            elif key == 'infos':
                frame_dict[key] = value[t]
        return Frame(**frame_dict)

    def __setitem__(self, t: int, frame: Frame) -> None:
        assert not self._empty, "Episode is empty."
        assert t < len(self), "Index out of range. Episode length is {len(self)} but got {t} as index."
        for key, value in frame.items():
            if isinstance(value, np.ndarray):
                self._data[key][t] = value
            elif isinstance(value, dict):
                for sub_key,sub_value in value.items():
                    self._data[key][sub_key][t] = sub_value

    def __iter__(self) -> Iterator[Frame]:
        for t in range(len(self)):
            yield self[t]

    def __len__(self) -> int:
        return 0 if self._empty else self._data["terminated"].shape[0]
    
    def __repr__(self) -> str:
        return f"Episode(episode_id={self._episode_id}," + ", ".join([f"{key}={value}" for key, value in self._data.items()]) + ")"

    def get_id(self) -> str:
        return self._episode_id

    def add_frame(self, frame: Frame) -> None:
        if self._empty:
            self._empty = False
            for key, value in frame.items():
                if isinstance(value, np.ndarray):
                    self._data[key] = np.expand_dims(value, axis=0)
                elif isinstance(value, dict) and key != 'infos':
                    self._data[key] = {sub_key: np.expand_dims(sub_value, axis=0) for sub_key, sub_value in value.items()}
                elif key == 'infos':
                    self._data[key] = [value]
        else:
            for key, value in frame.items():
                if isinstance(value, np.ndarray):
                    self._data[key] = np.concatenate((self._data[key], np.expand_dims(value, axis=0)), axis=0)
                elif isinstance(value, dict) and key != 'infos':
                    for sub_key, sub_value in value.items():
                        self._data[key][sub_key] = np.concatenate((self._data[key][sub_key], np.expand_dims(sub_value, axis=0)), axis=0)
                elif key == 'infos':
                    self._data[key].append(value)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return flatten_dict(self._data)
    
    def from_dict(data: Dict[str, np.ndarray], episode_id: str = None) -> "BaseEpisode":
        episode = BaseEpisode(episode_id)
        episode._data = unflatten_dict(data)
        episode._empty = False
        return episode

class EpisodeRelabeler(ABC):

    """
    Abstract base class for an episode relabeler.
    """

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, episode: Episode):
        """
        Relabels an episode.

        Args:
            episode (dict): The episode to relabel.
        
        Returns:
            (dict): The relabeled episode.
        """
        raise NotImplementedError

class EpisodesDB(ABC):

    """
    Abstract base class for a database of episodes.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []

    @abstractmethod
    def __contains__(self, episode_id: str) -> bool:
        """
        Check if an episode is in the database.

        Args:
            episode_id (str): The ID of the episode.
        
        Returns:
            (bool): True if the episode is in the database, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, episode_id: Union[str, int, Tuple[str,Any]]) -> Union[Episode,dict[str,np.ndarray]]:
        """
        Get an episode from the database given its ID.

        Args:
            episode_id (Union[str, int, Tuple[str,Any]]): The ID of the episode.

        Returns:
            (Union[Episode,dict[str, np.ndarray]]): The episode given its ID (eventually as a dictionary of arrays).
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Episode]:
        """
        Iterate over the episodes in the database.

        Returns:
            (Iterator[Episode]): An iterator over the episodes in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of episodes in the database.

        Returns:
            (int): The number of episodes in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Get a string representation of the database.

        Returns:
            (str): A string representation of the database.
        """
        raise NotImplementedError

    @abstractmethod
    def add_episode(self, episode: Episode) -> None:
        """
        Add an episode to the database.

        Args:
            episode (Episode): The episode to add.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_episode(self, episode_id: str) -> None:
        """
        Delete an episode from the database.

        Args:
            episode_id (str): The ID of the episode to delete.
        
        Returns:
            (None): Nothing.
        """
        raise NotImplementedError

    @abstractmethod
    def pop(self, episode_id: str) -> Episode:
        """
        Pop an episode from the database given its ID and return it.

        Args:
            episode_id (str): The ID of the episode to pop.
        
        Returns:
            (Episode): The popped episode.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ids(self) -> List[str]:
        """
        Get the IDs of the episodes in the database.

        Returns:
            (List[str]): The IDs of the episodes in the database.
        """
        raise NotImplementedError

class Sampler(ABC):

    """
    Abstract base class for a sampler of frames (or episodes) from a database of episodes.
    """
    
    @abstractmethod
    def sample_batch(self) -> Union[Dict[str, np.ndarray], None]:
        """
        Sample a batch of frames (or episodes) from the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of elements in the dataset.
        """
        raise NotImplementedError

class SamplerMulti(ABC):

    """
    Abstract base class for a sampler of frames (or episodes) from multiple databases of episodes.

    Args:
        episodes_dbs (List[EpisodesDB]): The list of databases of episodes.
        n_episodes (int): The number of episodes to sample. Defaults to None.
        context_size (int): The context size. Defaults to 0.
        padding_size_begin (int): The padding size at the beginning of the episodes. Defaults to 0.
        padding_size_end (int): The padding size at the end of the episodes. Defaults to 0.
        padding_value_begin (float): The padding value at the beginning of the episodes. Defaults to 0.
        padding_value_end (float): The padding value at the end of the episodes. Defaults to 0.
        reward_scalew (float): The reward scale weight. Defaults to 1.0.
        reward_scale_b (float): The reward scale bias. Defaults to 0.0.
    """

    def __init__(
        self,
        episodes_dbs: List[EpisodesDB],
        n_episodes: int = None,
        batch_size: int = 1,
        context_size: int = 0,
        padding_size_begin: int = 0,
        padding_size_end: int = 0,
        padding_value_begin: float = 0,
        padding_value_end: float = 0,
        reward_scale_w: float = 1.0,
        reward_scale_b: float = 0.0) -> None:

        # general parameters
        assert padding_size_begin >= context_size, "Padding size at the beginning of the episodes must be greater than or equal to the context size."
        self._episodes_dbs: List[EpisodesDB] = episodes_dbs
        self._n_episodes: List[int] = [n_episodes] * len(self._episodes_dbs)
        self._batch_size: int = batch_size
        self._context_size: int = context_size
        self._padding_size_begin: int = padding_size_begin
        self._padding_size_end: int = padding_size_end
        self._padding_value_begin: float = padding_value_begin
        self._padding_value_end: float = padding_value_end
        self._reward_scale_w: float = reward_scale_w
        self._reward_scale_b: float = reward_scale_b

        # episode ids
        if not self._n_episodes[0] is None: 
            self._n_episodes = [len(episodes_db) for episodes_db in self._episodes_dbs]
        
        for i,episodes_db in enumerate(self._episodes_dbs):
            assert self._n_episodes[i] <= len(episodes_db), "Number of episodes to sample is greater than the number of episodes in the database {}.".format(episodes_db)
        
        self._episodes_ids = [rd.sample(episodes_db.get_ids(), n_episodes) for episodes_db,n_episodes in zip(self._episodes_dbs,self._n_episodes)]
        self._episodes_lengths = [{episode_id: len(episodes_db[episode_id]) for episode_id in episodes_ids} for episodes_db,episodes_ids in zip(self._episodes_dbs,self._episodes_ids)]
    
    @abstractmethod
    def sample_batch(self) -> Union[Dict[str, np.ndarray], None]:
        """
        Sample a batch of frames (or episodes) from the database.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of elements in the dataset.
        """
        raise NotImplementedError