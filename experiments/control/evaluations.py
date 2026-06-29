import os
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict
from itertools import product
from omegaconf import DictConfig
from sciql.core.agent import Agent
from sciql.core.logger import Logger
from sciql.core.evaluation import Evaluation
from sciql.utils.labels import LabelsDistribution
from sciql.utils.gating import get_function
from sciql.utils.videos import save_episodes_videos
from sciql.utils.imports import instantiate_class, get_class, get_arguments
from sciql.utils.return_bounds import RETURN_BOUNDS

def compute_labels(episodes, env, label_names):
    """
    Takes the episodes and returns the concatenation
    of the label lists of each episodes, along with
    the sum of the length of each episode.

    Args:
        episodes (Episode): The list of the episodes.
        env (Env): The environment.
        label_names (List): The list of the labels names to compute.
    Returns:
        labels (Dict): The dictionnary of keys the names of the labels and
        of value the concatenation of the lists of the corresponding labels
        for all the episodes. e.g. {'speed_label': [label_0, ..., label_T, label_0, ...]}.
        episodes_dicts (List): List of the episodes dicts of the given episodes with added
        labels, e.g. [{'observation': [...], 'speed_label': [...]}].
        global_len (int): The sum of the length of all the episodes.
        global_labels (List): The list of the joint_labels, represented as lists.
    """
    labels, episode_dicts = {}, []
    global_len = 0
    for episode in episodes:
        global_len += len(episode)
        episode_dict = episode.to_dict()
        for label_name in label_names:
            label = env.labels[label_name]
            label_dict = label(episode)
            for k, v in label_dict.items():
                episode_dict[k] = np.array(v)
                if k in labels:
                    labels[k].extend(v)
                else:
                    labels[k] = v
        episode_dicts.append(episode_dict)
    global_labels = [[labels[k][t] for k in labels.keys()] for t in range(global_len)]
    return labels, episode_dicts, global_len, global_labels

def compute_label_percentages(labels, global_len, env):
    """
    Compute the percentage of transitions given a certain
    label.

    Args:
        labels (Dict): The dictionnary of keys the names of the labels and
        of value the concatenation of the lists of the corresponding labels
        for all the episodes. e.g. {'speed_label': [label_0, ..., label_T, label_0, ...]}
        global_len (int): The sum of the length of all the episodes.
        global_labels (List): The list of the joint_labels, represented as lists.
    Returns:
        percentages (Dict): The dictionnary of keys the criteria_names, and value the dictionnaries
        {'label_number': p(label_number) within data}.
    """
    percentages = {name: {str(i): 0 for i in range(env.labels[name].num_labels)} for name in labels.keys() if name != 'global_label'}
    label_ranges = [range(env.labels[name].num_labels) for name in labels.keys() if name != 'global_label']
    combinations = list(product(*label_ranges))
    percentages['global_label'] = {str(list(combination)): 0 for combination in combinations}
    for k, v in labels.items():
        for label in v:
            label = str(label)
            if label in percentages[k]:
                percentages[k][label] += 1
            else:
                print(k, label, percentages[k])
                percentages[k][label] = 1 
        for label in percentages[k].keys():
            percentages[k][label] /= global_len
    return percentages

class Control_Evaluation(Evaluation):

    def __init__(self, evaluation_cfg: DictConfig):
        self.name = 'control_evaluation'
        self.evaluation_cfg = evaluation_cfg
    
    def launch(self, agent: Agent, infos: Dict, logger: Logger):
        
        # Get used labels infos
        env = instantiate_class(self.evaluation_cfg.env)

        # List of the names of the criterion to put in conditioning
        eval_labels_names = list(self.evaluation_cfg.eval_labels_names)

        # List of labelers to put in conditioning
        eval_labels = [env.labels[name] for name in eval_labels_names]

        # List of the total number of promptable labels in the criterion, as there are labels for 'undefined' styles.
        eval_labels_numbers = [label.get_promptable_labels_number() for label in eval_labels]

        # List of the lists of labels to evaluate for each criterion
        assert len(eval_labels_names) == len(self.evaluation_cfg.eval_labels_idxs), 'Must have same length'
        eval_labels_idxs = []
        for i, (label, idxs) in enumerate(zip(eval_labels, self.evaluation_cfg.eval_labels_idxs)):
            if idxs is None:
                # Take all promptable labels
                idxs = np.arange(eval_labels_numbers[i])
            else:
                # Verify that all the asked labels are infact promptable
                assert all([label.is_promptable_idx(idx) for idx in idxs]), "All label idxs must be promptable."
            eval_labels_idxs.append(np.array(idxs))

        # List of the combinations of each labels to evaluate
        eval_labels_combinations = np.array(list(itertools.product(*eval_labels_idxs)), dtype=np.int32)

        # Print infos
        df_infos = pd.DataFrame({"label_name": eval_labels_names, "eval_labels_idxs": eval_labels_idxs})
        pd.set_option("display.max_colwidth", None)
        print(df_infos.to_string(index=False))

        # Compute episodes
        for labels_combination in tqdm(eval_labels_combinations, desc=f"Evaluating all labels combinations"):

            # Set experiment name according to the conditioning label combination
            agent_reset_args = {'labels': labels_combination}
            exp_name = f'{self.name}_label_{labels_combination}'

            # Get episodes by conditioning on the label combination
            episodes, videos = env.generate_episodes(
                agent=agent,
                task=self.evaluation_cfg.task_name,
                seed=self.evaluation_cfg.seed,
                n_episodes=self.evaluation_cfg.n_episodes,
                n_video_episodes=self.evaluation_cfg.n_video_episodes,
                agent_reset_args=agent_reset_args,
                agent_eval_mode_args=self.evaluation_cfg.agent_eval_mode_args,
                verbose=self.evaluation_cfg.verbose,
                render=self.evaluation_cfg.render
            )

            # Save videos
            if len(videos) > 0:
                save_episodes_videos(videos, os.path.join(self.evaluation_cfg.video_path, f'videos_{infos["gradient_step"]}', exp_name))

            # Compute labels distribution, e.g. p(criterion_0=1|criterion_1=2)
            labels, episodes_dicts, global_len, global_labels = compute_labels(episodes, env, eval_labels_names)
            
            # Compute the episodes infos, which corresponds to the percentage
            # of expertise, e.g. how close is the episode towards expertise
            # of the task or the style.
            episodes_infos = []
            for episode_dict in episodes_dicts:
                episode_infos = {}

                # Marginals labels infos
                for i_label, label_name in enumerate(eval_labels_names):
                    # Compute instead the indicators
                    R_min, R_max = 0.0, len(episode_dict[label_name])
                    R_episode = np.sum((episode_dict[label_name] == labels_combination[i_label]), dtype=np.float32)
                    episode_infos[f'{label_name}/regret'] = (R_max - R_episode).copy().item()
                    episode_infos[f'{label_name}/alignment'] = ((R_episode - R_min) / (R_max - R_min)).copy().item()

                # Task infos
                R_min, R_max = RETURN_BOUNDS[self.evaluation_cfg.task_name]
                R_episode = np.sum(episode_dict['reward'], dtype=np.float32)
                episode_infos['reward/regret'] = (R_max - R_episode).copy().item()
                episode_infos['reward/alignment'] = ((R_episode - R_min) / (R_max - R_min)).copy().item()
                
                # Joint labels infos
                # Compute instead the indicators
                R_min, R_max = 0.0, len(episode_dict[eval_labels_names[0]])
                joint_labels = np.stack([episode_dict[label_name] for label_name in eval_labels_names], axis=-1)
                R_episode = np.sum(np.all(joint_labels == labels_combination, axis=1), dtype=np.float32)
                episode_infos[f'joint_label/regret'] = (R_max - R_episode).copy().item()
                episode_infos[f'joint_label/alignment'] = ((R_episode - R_min) / (R_max - R_min)).copy().item()

                episodes_infos.append(episode_infos)
            
            # Extract criteria regrets
            regret_keys = list([k for k in episodes_infos[0].keys() if 'regret' in k])
            regrets = {k: np.array([episode_infos[k] for episode_infos in episodes_infos]).mean() for k in regret_keys}

            # Extract criteria alignments
            alignment_keys = list([k for k in episodes_infos[0].keys() if 'alignment' in k])
            alignments = {k: np.array([episode_infos[k] for episode_infos in episodes_infos]).mean() for k in alignment_keys}
            
            # Compute preferences alignments

            # Translate the joint_label term in the expression
            preference_alignments = {}
            for raw_eval_tree in ['joint_label', 'joint_label > reward', 'reward > joint_label']:
                eval_tree = raw_eval_tree.replace('joint_label', '*'.join(eval_labels_names))
                fn = get_function(
                    x_list=eval_labels_names, 
                    y_list=['reward'], 
                    operation_repr=eval_tree,
                    mode="discrete"
                )
                marginal_alignements = np.array([v for k, v in alignments.items() if k not in ['reward/alignment', 'joint_label/alignment']])
                reward_alignement = np.array([alignments['reward/alignment']])
                preference_alignment = fn(
                    x=marginal_alignements,  
                    y=reward_alignement
                )
                preference_alignments[f'preference_{raw_eval_tree}/alignment'] = preference_alignment
            for k, v in preference_alignments.items():
                alignments[k] = v
            
            # Log
            for k, v in alignments.items(): logger.add_scalar(f'{exp_name}/{k}', v, infos['gradient_step'])
            for k, v in regrets.items(): logger.add_scalar(f'{exp_name}/{k}', v, infos['gradient_step'])
            
            # Compute per episodes statistics
            episodes_statistics_lists = {}
            for episode in episodes:
                for statistic_name, statistic in env.statistics.items():
                    if statistic_name in set(['return', 'discounted_return', 'normalized_return', 'length'] + self.evaluation_cfg.statistics):
                        stat_dict = statistic(episode)
                        for k, v in stat_dict.items():
                            if k in episodes_statistics_lists:
                                episodes_statistics_lists[k].append(v)
                            else:
                                episodes_statistics_lists[k] = [v]
            
            # Compute and log metrics
            episodes_statistics = {}
            for k, v in episodes_statistics_lists.items():
                episodes_statistics[k] = (np.mean(episodes_statistics_lists[k]), np.std(episodes_statistics_lists[k]))
                logger.add_scalar(f'{exp_name}/{k}', episodes_statistics[k][0], infos['gradient_step'])
        
        return episodes_statistics
    