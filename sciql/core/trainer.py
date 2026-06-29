from abc import ABC, abstractmethod
from omegaconf import DictConfig, ListConfig
from sciql.core.agent import Agent, AgentsDB
from sciql.core.logger import Logger

class SingleLearningTrainer(ABC):


    """
    Absract class to define a learning algorithm.
    """

    @abstractmethod
    def train(
        algo_cfg: DictConfig,
        agent_cfg: DictConfig,
        agents_db: AgentsDB,
        logger: Logger,
        verbose: bool) -> Agent:
        """
        Executes the training loop for a learning algorithm.

        Args:
            algo_cfg (DictConfig): Configuration for the learning algorithm.
            task_cfg (DictConfig): Configuration for the task.
            agent_cfg (DictConfig): Configuration for the agent.
            agents_db (AgentsDB): A database to store agents.
            logger (Logger): A logger to log information.
            verbose (bool): Whether to print detailed information during training.

        Returns:
            (Agent): The trained agent.
        """

        # Seeding (set seeds)

        # Prepare dataset and dataloader (for online it would also be here that you initialize the environment and replay buffer)

        # Prepare agent (initialize the agent)

        # Restore agent if needed (call agent checkpoint from data)

        # Prepare evaluation (Initialize the evaluation policy if needed)

        # Train agent
        # for batch in ...:

            # Update agent (for online it would be here that you step the environment)

            # Log metrics (in the logger)

            # Evaluate agent

            # Save agent
        
        # Return and exit
        raise NotImplementedError

class ContinualLearningTrainer(ABC):

    """
    Abstract base class to define a continual learning algorithm for an identified single task (over a sequence of tasks).
    """

    @abstractmethod
    def run(
        algo_cfg: DictConfig,
        task_idx: int,
        tasks_cfg: ListConfig,
        agent_cfg: DictConfig,
        previous_agents_db: AgentsDB,
        current_agents_db: AgentsDB,
        logger: Logger,
        verbose: bool) -> Agent:
        """
        Runs the training algorithm.

        Args:
            algo_cfg (DictConfig): Configuration for the learning algorithm.
            task_idx (int): The index of the current task.
            tasks_cfg (ListConfig): Configuration for the tasks.
            agent_cfg (DictConfig): Configuration for the agent.
            previous_agents_db (AgentsDB): A database to store agents from the previous task.
            current_agents_db (AgentsDB): A database to store agents for the current task.
            logger (Logger): A logger to log information.
            verbose (bool): Whether to print detailed information during training.

        Returns:
            (Agent): The trained agent.
        """
        raise NotImplementedError
