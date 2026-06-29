import os
import shutil
from filelock import FileLock
from typing import List, Optional, Dict, Type
from sciql.core.agent import Agent, AgentsDB

class JaxAgentsDB(AgentsDB):
    """
    A thread-safe and process-safe database for storing and retrieving
    self-contained JAX agents like BC_Agent.

    This class manages agent checkpoints on the filesystem, delegating the
    actual serialization and deserialization to the agent's own `save` and `load` methods.
    """

    def __init__(self, directory: str) -> None:
        """
        Initializes the agent database.

        Args:
            directory (str): The root directory where all agent checkpoints will be stored.
        """
        self._directory = os.path.abspath(directory)
        self._lock_path = os.path.join(self._directory, ".db.lock")

        # Ensure the directory exists so we can create the lock file inside it.
        os.makedirs(self._directory, exist_ok=True)

    def _scan_dir(self) -> Dict[str, List[int]]:
        """
        Scans the directory to build a map of agent IDs to their saved stages.
        The result is cached to avoid repeated disk I/O. This operation is thread/process-safe.
        """
        if hasattr(self, '_cached_dir_scan'):
            return self._cached_dir_scan

        with FileLock(self._lock_path):
            agent_stages_map: Dict[str, List[int]] = {}
            for fname in os.listdir(self._directory):
                if '___' not in fname:
                    continue
                try:
                    agent_id, stage_str = fname.rsplit('___', 1)
                    stage = int(stage_str)
                    if agent_id not in agent_stages_map:
                        agent_stages_map[agent_id] = []
                    agent_stages_map[agent_id].append(stage)
                except (ValueError, IndexError):
                    continue
        
        self._cached_dir_scan = agent_stages_map
        return agent_stages_map

    def _clear_cache(self):
        """Invalidates the directory scan cache. Must be called after any write/delete."""
        if hasattr(self, '_cached_dir_scan'):
            del self._cached_dir_scan

    def get_ids(self) -> List[str]:
        """Returns a list of all unique agent IDs in the database."""
        return list(self._scan_dir().keys())

    def stages(self, agent_id: str) -> List[int]:
        """Returns a sorted list of all saved stages for a given agent ID."""
        return sorted(self._scan_dir().get(agent_id, []))
    
    def __contains__(self, agent_id: str) -> bool:
        """Checks if an agent ID exists in the database."""
        return agent_id in self._scan_dir()

    def __repr__(self) -> str:
        """Provides a human-readable representation of the database."""
        num_agents = len(self.get_ids())
        return f"JaxAgentsDB(directory='{self._directory}') with {num_agents} agents."
        
    def add_agent(self, agent: Agent, agent_id: str, agent_stage: int) -> None:
        """
        Saves an agent's state to the database at a specific stage by calling its .save() method.

        Args:
            agent (Agent): The agent instance to save. Must have a `save(path)` method.
            agent_id (str): A unique ID for the agent. Cannot contain '___'.
            agent_stage (int): The training stage (e.g., step number) of the agent.
        """
        if "___" in agent_id:
            raise ValueError("Agent ID cannot contain '___'.")
        
        with FileLock(self._lock_path):
            path = os.path.join(self._directory, f"{agent_id}___{agent_stage}")
            # Delegate the saving logic to the agent itself.
            agent.save(path)
            self._clear_cache()

    def get(self, agent_id: str, agent_stage: int, agent_class: Type[Agent]) -> Agent:
        """
        Restores an agent with the specified ID and stage by calling its .load() classmethod.

        Args:
            agent_id (str): The agent ID.
            agent_stage (int): The agent stage to load.
            agent_class (Type[Agent]): The class of the agent to load (e.g., `BC_Agent`).
                                       Must have a `load(path)` classmethod.

        Returns:
            The restored agent instance.
        """
        path = os.path.join(self._directory, f"{agent_id}___{agent_stage}")
        
        with FileLock(self._lock_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Agent '{agent_id}' at stage {agent_stage} not found in '{self._directory}'.")
            
            # Delegate the loading logic to the agent class itself.
            return agent_class.load(path)

    def delete_agent(self, agent_id: str, agent_stage: Optional[int] = None) -> None:
        """Deletes an agent's checkpoint(s). If agent_stage is None, deletes all stages."""
        stages_to_delete = [agent_stage] if agent_stage is not None else self.stages(agent_id)
        if not stages_to_delete:
            return

        with FileLock(self._lock_path):
            for stage in stages_to_delete:
                dir_path = os.path.join(self._directory, f"{agent_id}___{stage}")
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            self._clear_cache()

    def get_first_stage(self, agent_id: str, agent_class: Type[Agent]) -> Agent:
        """A convenience method to get the earliest saved stage of the specified agent."""
        all_stages = self.stages(agent_id)
        if not all_stages:
            raise FileNotFoundError(f"Agent '{agent_id}' not found in database.")
        # The `stages` method already sorts, so the first element is the earliest stage.
        earliest_stage = all_stages[0]
        return self.get(agent_id, earliest_stage, agent_class)

    def get_last_stage(self, agent_id: str, agent_class: Type[Agent]) -> Agent:
        """A convenience method to get the latest saved stage of the specified agent."""
        all_stages = self.stages(agent_id)
        if not all_stages:
            raise FileNotFoundError(f"Agent '{agent_id}' not found.")
        return self.get(agent_id, max(all_stages), agent_class)