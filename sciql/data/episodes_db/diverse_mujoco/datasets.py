import os
import numpy as np
from collections import defaultdict

def load_diverse_mujoco_dataset(
        dataset_dir,
        archetypes: list = None,
        ob_dtype=np.float32,
        action_dtype=np.float32,
        compact_dataset=False
    ):
    """Load and CONCATENATE multiple OGBench-style MuJoCo datasets.

    Each <archetype>.npz must contain:
      'observations', 'actions', 'rewards', 'terminated', 'truncated'

    Returns:
      If compact_dataset:
        {'observations','actions','rewards','terminated','truncated','valids'}
      else:
        {'observations','actions','rewards','terminated','truncated','next_observations'}
      where in the non-compact case, terminated/truncated are aligned with (s_t, a_t, s_{t+1}).
    """
    # Discover archetypes if not provided
    if archetypes is None:
        archetypes = [
            os.path.splitext(fn)[0]
            for fn in sorted(os.listdir(dataset_dir))
            if fn.endswith(".npz") and ('val' not in fn)
        ]

    # Accumulators
    obs_list, act_list, rew_list = [], [], []
    term_list, trunc_list = [], []
    next_obs_list = []   # only for compact_dataset == False
    valids_list = []     # only for compact_dataset == True
    infos_lists = defaultdict(list)

    for archetype in archetypes:
        dataset_path = os.path.join(dataset_dir, f"{archetype}.npz")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Missing dataset file: {dataset_path}")

        file = np.load(dataset_path)

        # Load & cast
        observations = file['observations'][...].astype(ob_dtype)
        actions      = file['actions'][...].astype(action_dtype)
        rewards      = file['rewards'][...].astype(np.float32)
        terminated   = file['terminated'][...].astype(np.float32)  # keep float32; change to bool if you prefer
        truncated    = file['truncated'][...].astype(np.float32)
        infos = {k: v[...].astype(np.float32) for k, v in file.items() if 'infos' in k}
        
        dones_raw = (terminated.astype(bool) | truncated.astype(bool))
        dones_raw = dones_raw.astype(np.float32)

        if compact_dataset:
            # Keep per-step termination signals as recorded.
            # valids marks indices with a valid next state within the same trajectory.
            valids = 1.0 - dones_raw

            obs_list.append(observations)
            act_list.append(actions)
            rew_list.append(rewards)
            term_list.append(terminated)
            trunc_list.append(truncated)
            for k in infos.keys(): infos_lists[k].append(infos[k])
            valids_list.append(valids.astype(np.float32))
        else:
            # Build aligned transitions (s_t, a_t, s_{t+1}) without crossing boundaries.
            ob_mask = (1.0 - dones_raw).astype(bool)       # only steps that have a valid s_{t+1}
            next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
            next_observations = observations[next_ob_mask]

            observations_kept = observations[ob_mask]
            actions_kept      = actions[ob_mask]
            rewards_kept      = rewards[ob_mask]

            # Shift termination signals so they refer to s_{t+1} in the kept transitions.
            shifted_terminated = np.concatenate([terminated[1:], [1.0]])
            shifted_truncated  = np.concatenate([truncated[1:],  [1.0]])

            terminated_kept = shifted_terminated[ob_mask].astype(np.float32)
            truncated_kept  = shifted_truncated[ob_mask].astype(np.float32)

            obs_list.append(observations_kept)
            act_list.append(actions_kept)
            rew_list.append(rewards_kept)
            for k in infos.keys(): infos_lists[k].append(infos[k][ob_mask])
            term_list.append(terminated_kept)
            trunc_list.append(truncated_kept)
            next_obs_list.append(next_observations.astype(ob_dtype))

    # Concatenate across archetypes
    dataset = dict()
    dataset['observations'] = np.concatenate(obs_list, axis=0) if obs_list else np.empty((0,), dtype=ob_dtype)
    dataset['actions']      = np.concatenate(act_list, axis=0) if act_list else np.empty((0,), dtype=action_dtype)
    dataset['rewards']      = np.concatenate(rew_list, axis=0) if rew_list else np.empty((0,), dtype=np.float32)
    dataset['terminated']   = np.concatenate(term_list, axis=0) if term_list else np.empty((0,), dtype=np.float32)
    dataset['truncated']    = np.concatenate(trunc_list, axis=0) if trunc_list else np.empty((0,), dtype=np.float32)
    for k in infos_lists.keys(): dataset[k] =  np.concatenate(infos_lists[k], axis=0) if infos_lists[k] else np.empty((0,), dtype=np.float32)

    print(
        f"""
        Loaded archetypes: {archetypes} in {dataset['observations'].shape[0]} steps.
        """
    )

    if compact_dataset:
        dataset['valids'] = np.concatenate(valids_list, axis=0) if valids_list else np.empty((0,), dtype=np.float32)
    else:
        dataset['next_observations'] = (
            np.concatenate(next_obs_list, axis=0) if next_obs_list else np.empty((0,), dtype=ob_dtype)
        )

    # Drop speed_z if needed
    dataset['observations'] = dataset['observations'][:,:17]
    dataset['next_observations'] = dataset['next_observations'][:,:17]

    return dataset
