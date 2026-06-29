import numpy as np
from tqdm import tqdm
from sciql.core.data import EpisodesDB

def episodes_db_to_numpy_dataset(
        episodes_db: EpisodesDB, 
        episodes_relabelers: list = [], 
        keys: list = ['observations', 'actions', 'terminals', 'timeouts', 'dones']
    ):
    """
    From an episodes_reader contraining self.episodes, which is a list of N episodes as dictionaries of the minimal form: 
    {
        'observations': np.ndarray[T, ...], T the episode length         -> [s_0, s_1, s_2, ..., s_T-1] (the sequence of observations)
        'actions': np.ndarray[T, ...], T the episode length              -> [a_0, a_1, a_2, ..., a_T-1] (the sequence of actions)
        'terminals': np.ndarray[T, ...], T the episode length            -> [0, 0, ..., 0 or 1]         (if the state of next_observations is a real terminal state)
        'timeouts': np.ndarray[T, ...], T the episode length             -> [0, 0, ..., 0 or 1]         (if the state of next_observations as been timeout)
        'dones': np.ndarray[T, ...], T the episode length                -> 'terminals' or 'timeouts'   (if the state of next_observations is the last of the recorded sequence)
    }
    Create the dataset of the minimal form:
    {
        'observations': np.ndarray[N*T, ...], T the episode length
        'actions': np.ndarray[N*T, ...], T the episode length
        'terminals': np.ndarray[N*T, ...], T the episode length
        'timeouts': np.ndarray[N*T, ...], T the episode length
        'dones': np.ndarray[N*T, ...], T the episode length
    }

    Args:
        episodes_db (EpisodesDB): EpisodesDB to convert.
        episodes_relabelers (): 

    """
    
    # Get dataset keys
    keys = set(keys)
    relabelers_keys = set([])
    for episodes_relabeler in episodes_relabelers:
        relabelers_keys.update(episodes_relabeler.keys)
    keys.update(relabelers_keys)
    
    # Build dataset
    dataset = {k:[] for k in keys}
    for id in tqdm(episodes_db.get_ids(), desc='Building numpy dataset'):
        
        # Get episode
        episode = episodes_db[id]

        # Relabeling of episode
        for episodes_relabeler in episodes_relabelers:
            episode = episodes_relabeler(episode)

        # Add 
        for k in keys:
            dataset[k].append(episode[k])

    for k in keys:
        dataset[k] = np.concatenate(dataset[k])

    return dataset, relabelers_keys


def numpy_dataset_to_ogbench_dataset(
        dataset: dict, 
        relabelers_keys: list[str] = [], 
        compact_dataset: bool = True,
        obs_keys: list[str] = [''],
        action_keys: list[str] = ['']
    ):
    """
    Turns the numpy dataset into an ogbench dataset.
    """
    # Example:
    # Assume each trajectory has length 4, and (s0, a0, s1), (s1, a1, s2), (s2, a2, s3), (s3, a3, s4) are transition
    # tuples. Note that (s4, a4, s0) is *not* a valid transition tuple, and a4 does not have a corresponding next state.
    # At this point, `dataset` loaded from the file has the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  1, ...]

    if compact_dataset:
        # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
        # `next_observations[t]` by using `observations[t + 1]`.
        # Our goal is to have the following structure:
        #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
        # -------------------------------------------------------------
        # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
        # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
        # 'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
        # 'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
    else:
        # Regular dataset: Generate `next_observations` by shifting `observations`.
        # Our goal is to have the following structure:
        #                       |<- traj 1 ->|  |<- traj 2 ->|  ...
        # ----------------------------------------------------------
        # 'observations'     : [s0, s1, s2, s3, s0, s1, s2, s3, ...]
        # 'actions'          : [a0, a1, a2, a3, a0, a1, a2, a3, ...]
        # 'next_observations': [s1, s2, s3, s4, s1, s2, s3, s4, ...]
        # 'terminals'        : [ 0,  0,  0,  1,  0,  0,  0,  1, ...]

        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])

        # Get observations
        for k in obs_keys:
            dataset[f'next_observations{k}'] = dataset[f'observations{k}'][next_ob_mask]
            dataset[f'observations{k}'] = dataset[f'observations{k}'][ob_mask]

        # Get actions
        for k in action_keys:
            dataset[f'actions{k}'] = dataset[f'actions{k}'][ob_mask]

        # Get relabeled
        for k in relabelers_keys:
            dataset[k] = dataset[k][ob_mask]
        
        # Get infos
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)
        dataset['timeouts'] = dataset['terminals']
        dataset['dones'] = dataset['terminals']
    
    return dataset