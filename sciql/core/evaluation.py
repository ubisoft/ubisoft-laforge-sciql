from abc import ABC, abstractmethod
from omegaconf import DictConfig, ListConfig
from sciql.core.agent import Agent, AgentsDB
from sciql.core.logger import Logger
from typing import Dict, Union, Optional, Any, List

class Evaluation(ABC):
    
    """
    Abstract method for an evaluation. An evaluation takes an agent, infos 
    and logger and performs analyses and logs and returns the results.
    
    Args:
            evaluation_cfg (DictConfig): Configuration of the evaluation.
    """
    @abstractmethod
    def __init__(self, evaluation_cfg: DictConfig):
        self.name = 'rl_evaluation'
        self.evaluation_cfg = evaluation_cfg

    @abstractmethod
    def launch(self, agent: Agent, infos: Dict, logger: Logger) -> Dict:
        """
        Launches the evaluation.

        Args:
            agent (Agent): Agent to evaluate.
            infos (Dict): Informations about the agent, e.g. the gradient step.
            logger (Logger): Logger to log informations and statistics.

        Returns:
            (Dict): The resulting statistics of the evaluation.
        """
        raise NotImplementedError
        