import os
import pickle
import glob
import uuid
import numpy as np
from typing import Dict, Union, Optional, Any, List
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from sciql.core.data import EpisodesDB, Episode
from sciql.utils.paths import DATASETS_PATH

class Traj2D_EpisodesDB(EpisodesDB):
    """
    A database of episodes stored as individual pickle files in a directory.
    Episode IDs are random strings (UUIDs by default if not provided).
    Integer indexing refers to the i-th episode based on lexicographical sort of these IDs.
    """

    def __init__(self, directory: str, prefix: str = "episode", extension: str = "episode"):
        super().__init__()
        self.directory = os.path.join(DATASETS_PATH, directory)
        print(self.directory)
        self.prefix = prefix
        self.extension = extension
        os.makedirs(self.directory, exist_ok=True)
        self._load_existing_ids() # This sorts self._ids

    def _get_filepath(self, episode_id: str) -> str:
        return os.path.join(self.directory, f"{self.prefix}_{episode_id}.{self.extension}")

    def _get_id_from_filename(self, filename: str) -> Optional[str]:
        base_name = os.path.basename(filename)
        if base_name.startswith(self.prefix + "_") and base_name.endswith("." + self.extension):
            id_part = base_name[len(self.prefix) + 1 : -(len(self.extension) + 1)]
            return id_part
        return None

    def _load_existing_ids(self) -> None:
        self._ids = []
        pattern = os.path.join(self.directory, f"{self.prefix}_*.{self.extension}")
        for filepath in glob.glob(pattern):
            episode_id = self._get_id_from_filename(os.path.basename(filepath))
            if episode_id:
                self._ids.append(episode_id)
        self._ids.sort() # Ensure IDs are lexicographically sorted

    def __contains__(self, episode_id: str) -> bool:
        return episode_id in self._ids

    def __getitem__(self, key: Union[str, int, Tuple[str, Any]]) -> Union[Episode, Dict[str, np.ndarray]]:
        episode_id_to_load: Optional[str] = None

        if isinstance(key, str):
            episode_id_to_load = key
        elif isinstance(key, int):
            if not (0 <= key < len(self._ids)):
                raise IndexError(f"Integer index {key} is out of range for {len(self._ids)} episodes.")
            episode_id_to_load = self._ids[key] # Get ID string from sorted list
        elif isinstance(key, tuple) and len(key) > 0 and isinstance(key[0], str):
            episode_id_to_load = key[0]
        else:
            raise TypeError(f"Key type {type(key)} not supported. Must be str, int, or Tuple[str, Any].")

        if episode_id_to_load is None:
             raise ValueError("Could not determine episode ID from key.")

        if episode_id_to_load not in self._ids:
            self._load_existing_ids() # Refresh if ID not found, maybe added externally
            if episode_id_to_load not in self._ids:
                 raise KeyError(f"Episode with ID '{episode_id_to_load}' not found in the database.")
        
        filepath = self._get_filepath(episode_id_to_load)
        try:
            with open(filepath, 'rb') as f:
                episode_data = pickle.load(f)
            
            if not isinstance(episode_data, Episode):
                if isinstance(episode_data, dict): return episode_data 
                raise TypeError(f"Loaded data for ID '{episode_id_to_load}' is not of type Episode or dict.")
            return episode_data
        except FileNotFoundError:
            self._load_existing_ids()
            raise KeyError(f"Episode file for ID '{episode_id_to_load}' not found (path: {filepath}).")
        except Exception as e:
            raise IOError(f"Error loading episode '{episode_id_to_load}' from '{filepath}': {e}")

    def __iter__(self) -> Iterator[Episode]:
        for episode_id in self._ids: # Iterates in sorted order of IDs
            try:
                loaded_item = self[episode_id]
                if isinstance(loaded_item, Episode):
                    yield loaded_item
                else:
                    print(f"Warning: Item for ID {episode_id} is a dict, not an Episode. Skipping in typed iteration.")
            except (KeyError, IOError, TypeError) as e:
                print(f"Warning: Could not load episode {episode_id} during iteration: {e}")
                continue

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"<PickleEpisodesDB directory='{self.directory}', count={len(self)}>"

    def add_episode(self, episode: Episode, episode_id: Optional[str] = None) -> None:
        if episode_id is None:
            # Generate a random string ID (UUID4 hex)
            # Loop to ensure uniqueness (extremely low probability of collision with UUID4)
            while True:
                generated_id = uuid.uuid4().hex 
                # Check against current _ids and also if a file might exist with this ID
                # (e.g. if _ids hasn't been reloaded recently and a file was added externally)
                # For simplicity here, we mostly rely on self._ids being up-to-date.
                # A robust check would be `not os.path.exists(self._get_filepath(generated_id))`
                if generated_id not in self._ids:
                    episode_id = generated_id
                    break
        elif not isinstance(episode_id, str):
            raise TypeError("episode_id must be a string or None.")

        if episode_id in self._ids and os.path.exists(self._get_filepath(episode_id)):
            print(f"Warning: Overwriting existing episode with ID '{episode_id}'.")
        
        filepath = self._get_filepath(episode_id)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(episode, f)
            if episode_id not in self._ids:
                self._ids.append(episode_id)
                self._ids.sort() # Ensure self._ids remains sorted after adding
        except Exception as e:
            raise IOError(f"Error saving episode '{episode_id}' to '{filepath}': {e}")

    def delete_episode(self, episode_id: str) -> None:
        if episode_id not in self._ids:
            self._load_existing_ids()
            if episode_id not in self._ids:
                 raise KeyError(f"Episode with ID '{episode_id}' not found for deletion.")
        
        filepath = self._get_filepath(episode_id)
        try:
            os.remove(filepath)
            self._ids.remove(episode_id) # No need to re-sort after removal from a sorted list
        except FileNotFoundError:
            if episode_id in self._ids: self._ids.remove(episode_id)
            print(f"Warning: Episode file for ID '{episode_id}' was already deleted from disk.")
        except Exception as e:
            raise IOError(f"Error deleting episode file '{filepath}': {e}")

    def pop(self, episode_id: str) -> Episode:
        episode_data = self[episode_id]
        if not isinstance(episode_data, Episode):
            raise TypeError(f"Item with ID '{episode_id}' is not an Episode, cannot pop as Episode.")
        self.delete_episode(episode_id)
        return episode_data

    def get_ids(self) -> List[str]:
        return list(self._ids) # self._ids is kept sorted