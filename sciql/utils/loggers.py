import numpy as np
import os
import pickle
import csv
import pandas as pd
from typing import Any, Dict, List, Union
from omegaconf import DictConfig, ListConfig
from sciql.core.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Union, Optional

def convert_cfg(data:Union[ListConfig,List,DictConfig,Dict,Any]) -> Union[List,Dict,Any]:
    """
    
    Convert a DictConfig or ListConfig object to a standard dictionary or list.

    Args:
        data (Union[ListConfig,List,DictConfig,Dict,Any]): The data to convert.
    
    Returns:
        Union[List,Dict,Any]: The converted data.
    """
    if isinstance(data,(ListConfig,List)): return [convert_cfg(e) for e in data]
    elif isinstance(data,(DictConfig,Dict)): return {k: convert_cfg(v) for k,v in data.items()}
    else: return data

class TensorBoardLogger(Logger):

    """
    A logger that logs to TensorBoard.

    Args:
        directory (str): The directory to log to.
        prefix (str): The prefix to add to all logs.
        max_cache_size (int): The maximum cache size before flushing.
    """

    def __init__(
        self, 
        directory: str, 
        prefix: str=None, 
        max_cache_size: int=1000, 
        log_to_csv: bool=False, 
        csv_names: List[str]=['training']
    ) -> None:
        
        self._directory: str = directory
        self._prefix: str = prefix + "/" if prefix else ""
        self._max_cache_size: int = max_cache_size
        os.makedirs(self._directory,exist_ok=True)
        self._logger: SummaryWriter = SummaryWriter(log_dir=self._directory)
        self._cache: List[Tuple[str,Union[np.ndarray,int,float,str],int]] = []

        self._log_to_csv = log_to_csv
        self._csv_files = []
        self._csv_writers =  []
        self._csv_names = csv_names
        if self._log_to_csv:
            for csv_name in csv_names:
                csv_path = os.path.join(self._directory, f"{csv_name}.csv")
                self._csv_files.append(open(csv_path, mode="w", newline=""))
                self._csv_writers.append(csv.writer(self._csv_files[-1]))
                self._csv_writers[-1].writerow(["name", "value", "epoch"])

    def log_params(self, params: Dict, directory: str=None) -> None:
        """
        Save the parameters to disk.

        Args:
            params (dict): The parameters to log.
            directory (str): The directory to log the parameters to. If None, log to the default directory. Default is None.
        
        Returns:
            (None): Nothing.
        """
        if directory is None: log_directory = self._directory
        else: log_directory = directory
        with open(f"{log_directory}/params.pickle","wb") as file:
            pickle.dump(convert_cfg(params),file)
    
    def add_histogram(self,tag:str,values:np.ndarray,step:int):
        """
        Log a histogram to TensorBoard.

        Args:
            tag (str): The tag for the histogram.
            values (np.ndarray): The values to log as a histogram.
            step (int): The step for the histogram.
        """
        _tag = self._prefix + tag
        self._logger.add_histogram(_tag, values, step)
        self._cache.append((_tag, values, step))
        if len(self._cache) > self._max_cache_size: self._flush()

    def add_image(self,tag:str,image:np.ndarray,step:int) -> None:
        """
        Log an image to TensorBoard.

        Args:
            tag (str): The tag for the image.
            step (int): The step for the image.
            image (np.ndarray): The image to add.
        
        Returns:
            (None): Nothing.
        """
        _tag = self._prefix + tag
        self._logger.add_image(_tag, image, step)
        self._cache.append((_tag, image, step))
        if len(self._cache) > self._max_cache_size: self._flush()

    def add_scalar(self, tag: str, value: Union[int,float], step:int) -> None:
        """
        Log a scalar to TensorBoard.

        Args:
            tag (str): The tag for the scalar.
            step (int): The step for the scalar.
            value (Union[int, float]): The value to add.
        
        Returns:
            (None): Nothing.
        """
        _tag = self._prefix + tag
        self._logger.add_scalar(_tag, value, step)
        self._cache.append((_tag, value, step))

        if self._log_to_csv and self._csv_writers:
            for i, csv_name in enumerate(self._csv_names):
                if csv_name in tag:
                    self._csv_writers[i].writerow([tag, value, step])
                    self._csv_files[i].flush()  # Ensure the data is written immediately
                    break

        if len(self._cache) > self._max_cache_size: self._flush()

    def add_text(self,tag:str,text:str,step:int) -> None:
        """
        Log text to TensorBoard.

        Args:
            tag (str): The tag for the text.
            step (int): The step for the text.
            text (str): The text to add.
        
        Returns:
            (None): Nothing.
        """
        _tag = self._prefix + tag
        self._logger.add_text(_tag, text, step)
        self._cache.append((_tag, text, step))
        if len(self._cache) > self._max_cache_size: self._flush()

    def close(self) -> None:
        """
        Close the logger.
        
        Returns:
            (None): Nothing.
        """
        self._flush()
        if not self._logger is None: 
            self._logger.close()
        if self._log_to_csv and self._csv_files:
            for csv_file in self._csv_files:
                csv_file.close()

    def _flush(self) -> None:
        """
        Flush the cache to disk.
        
        Returns:
            (None): Nothing.
        """
        if len(self._cache) > 0:
            with open(f"{self._logger.log_dir}/values.pickle","ab") as file:
                pickle.dump(self._cache,file)
            self._cache = []

    def __del__(self) -> None:
        """
        Destructor for the logger.
        
        Returns:
            (None): Nothing.
        """
        self.close()

    def delete_tf_events(self) -> None:
        """
        Delete the .tf.events file generated by TensorBoard.
        
        Returns:
            (None): Nothing.
        """
        for file in os.listdir(self._logger.log_dir):
            if "events.out.tfevents" in file:
                os.remove(f"{self._logger.log_dir}/{file}")

def load_log_dataframes(
    directory: str,
    csv_names: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Loads CSV log files from a directory into pandas DataFrames.

    This function is designed to read the data generated by the
    CSVTensorBoardLogger.

    Args:
        directory (str): The log directory where the CSV files are stored.
        csv_names (Optional[List[str]]): A list of specific CSV names to load
            (without the .csv extension, e.g., ['training', 'evaluation']).
            If None, the function will attempt to load all .csv files in the
            directory.

    Returns:
        A dictionary where keys are the CSV names and values are the
        corresponding pandas DataFrames. Returns an empty dictionary if the
        directory is not found.
    """
    loaded_data = {}

    if not os.path.isdir(directory):
        print(f"Error: Log directory not found at '{directory}'")
        return loaded_data

    target_files = []
    if csv_names is None:
        # If no names are specified, find all .csv files in the directory
        try:
            target_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        except FileNotFoundError:
            return loaded_data # Should be caught by the isdir check, but for safety
    else:
        # Use the provided names
        target_files = [f"{name}.csv" for name in csv_names]

    for filename in target_files:
        file_path = os.path.join(directory, filename)
        name_key = os.path.splitext(filename)[0]

        if not os.path.exists(file_path):
            print(f"Warning: CSV file not found at '{file_path}', skipping.")
            continue

        try:
            # The logger saves 'step' as the index, so we load it back with index_col=0.
            # If your index column has a name, use index_col='step_name'
            df = pd.read_csv(file_path)
            loaded_data[name_key] = df
            print(f"Successfully loaded '{file_path}'.")
        except Exception as e:
            print(f"Warning: Could not read or parse '{file_path}', skipping. Error: {e}")

    return loaded_data

class CSVTensorBoardLogger(Logger):
    """
    A logger that logs to TensorBoard and CSV files, with dynamic column creation.

    This logger uses pandas to manage CSV data, allowing for a flexible structure.
    When a new metric (key) is logged for the first time, a new column is
    automatically added to the corresponding CSV file. All previous steps (rows)
    for that new column are back-filled with null values (NaN).

    Each call to `add_scalar` immediately updates the CSV on disk.

    Args:
        directory (str): The directory to log to.
        prefix (str): The prefix to add to all TensorBoard tags.
        log_to_csv (bool): Whether to log scalar values to CSV files.
        csv_names (List[str]): Identifiers for grouping tags into CSV files.
                               (e.g., ['training', 'rl_evaluation'])
    """

    def __init__(
        self,
        directory: str,
        prefix: str = None,
        log_to_csv: bool = True,
        csv_names: List[str] = ['training', 'rl_evaluation']
    ) -> None:

        self._directory: str = directory
        self._prefix: str = prefix + "/" if prefix else ""
        os.makedirs(self._directory, exist_ok=True)

        # --- Tensorboard Setup ---
        self._logger: SummaryWriter = SummaryWriter(log_dir=self._directory)

        # --- CSV Setup with Pandas ---
        self._log_to_csv = log_to_csv
        self._csv_names = csv_names
        self._csv_paths: Dict[str, str] = {}
        # This dictionary will hold the pandas DataFrames in memory
        self._data_frames: Dict[str, pd.DataFrame] = {}

        if self._log_to_csv:
            for name in self._csv_names:
                csv_path = os.path.join(self._directory, f"{name}.csv")
                self._csv_paths[name] = csv_path

                # If the CSV already exists, load it into a DataFrame
                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                    try:
                        # Load existing data, using the first column ('step') as the index
                        df = pd.read_csv(csv_path, index_col=0)
                        self._data_frames[name] = df
                        print(f"Loaded existing data from '{name}.csv'. Columns: {df.columns.tolist()}")
                    except Exception as e:
                        print(f"Warning: Could not load existing CSV '{name}.csv'. "
                              f"It might be corrupted. Starting fresh. Error: {e}")
                        # If loading fails, start with an empty DataFrame
                        self._data_frames[name] = pd.DataFrame().rename_axis('step')
                else:
                    # If the file is new or empty, create an empty DataFrame
                    # We define 'step' as the index name from the beginning.
                    print(f"Creating new logger for '{name}.csv'.")
                    self._data_frames[name] = pd.DataFrame().rename_axis('step')


    def add_scalar(self, tag: str, value: Union[int, float], step: int) -> None:
        """
        Log a scalar to TensorBoard and directly to the corresponding CSV file.

        If the `tag` (column) does not exist, it is created. If the `step` (row)
        does not exist, it is also created. The value is placed at the
        intersection, and the file is immediately saved.

        Args:
            tag (str): The tag for the scalar (e.g., 'training/loss').
                       This becomes the column header.
            value (Union[int, float]): The value to log.
            step (int): The step for the scalar. This becomes the row index.
        """
        # --- Log to TensorBoard (unchanged) ---
        _tag_tb = self._prefix + tag
        try:
            self._logger.add_scalar(_tag_tb, value, step)
        except Exception as e:
            print(f"Warning: Could not log scalar to Tensorboard '{_tag_tb}' at step {step}: {e}")

        # --- Write directly to CSV using Pandas ---
        if self._log_to_csv:
            found_csv = False
            for name in self._csv_names:
                if name in tag:
                    try:
                        df = self._data_frames[name]
                        column_name = tag

                        # This is the core logic:
                        # Use .loc to assign the value. Pandas handles creating the
                        # row (index `step`) and/or the `column_name` if they
                        # don't exist, back-filling with NaN automatically.
                        df.loc[step, column_name] = float(value)

                        # Sort the DataFrame by step (index) to keep it ordered
                        df.sort_index(inplace=True)
                        
                        # Immediately write the entire updated DataFrame to disk.
                        # `index=True` ensures the 'step' index is saved as a column.
                        df.to_csv(self._csv_paths[name], index=True)

                    except Exception as e:
                        print(f"Error: Could not write to CSV for tag '{tag}' at step {step}. Error: {e}")

                    found_csv = True
                    break
            
            # if not found_csv:
            #     print(f"Warning: Tag '{tag}' did not match any known csv_name: {self._csv_names}")


    def flush(self) -> None:
        """
        Ensures all data is written to disk. In this implementation, data is
        written immediately in `add_scalar`, but this method provides a
        consistent interface and flushes the TensorBoard writer.
        """
        print("Flushing logger...")
        # For pandas-based logger, saving happens in add_scalar, but we can
        # re-save everything as a safety measure if needed.
        # for name, df in self._data_frames.items():
        #     df.to_csv(self._csv_paths[name], index=True)

        if self._logger is not None:
            self._logger.flush()
        print("Flush complete.")


    def close(self) -> None:
        """
        Flushes any final data and closes the TensorBoard writer.
        """
        print("Closing logger...")
        self.flush() # Ensure everything is written

        if self._logger is not None:
            try:
                self._logger.close()
                self._logger = None
            except Exception as e:
                print(f"Warning: Error closing TensorBoard writer: {e}")

        # Clear the in-memory dataframes
        self._data_frames = {}
        print("Logger closed.")


    def __del__(self) -> None:
        """ Destructor ensures close() is called. """
        if hasattr(self, '_logger') and self._logger is not None:
            self.close()

    # --- Other methods remain largely unchanged ---

    def log_params(self, params: Dict, directory: str = None) -> None:
        # (This method is independent of CSV logging and can remain the same)
        log_directory = directory if directory is not None else self._directory
        os.makedirs(log_directory, exist_ok=True)
        serializable_params = {k: str(v) if not isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))) else v for k, v in params.items()}
        try:
            with open(os.path.join(log_directory, "params.pickle"), "wb") as file:
                 pickle.dump(serializable_params, file)
        except Exception as e:
            print(f"Warning: Could not pickle params: {e}")
            with open(os.path.join(log_directory, "params.txt"), "w") as file:
                 for k, v in serializable_params.items():
                     file.write(f"{k}: {v}\n")

    def add_histogram(self,tag:str,values:np.ndarray,step:int):
        _tag = self._prefix + tag
        try:
            self._logger.add_histogram(_tag, values, step)
        except Exception as e:
            print(f"Warning: Could not log histogram '{_tag}' at step {step}: {e}")

    def add_image(self,tag:str,image:np.ndarray,step:int) -> None:
        _tag = self._prefix + tag
        try:
            self._logger.add_image(_tag, image, step, dataformats='HWC')
        except Exception as e:
             print(f"Warning: Could not log image '{_tag}' at step {step}: {e}")

    def add_text(self,tag:str,text:str,step:int) -> None:
        _tag = self._prefix + tag
        try:
            self._logger.add_text(_tag, text, step)
        except Exception as e:
            print(f"Warning: Could not log text '{_tag}' at step {step}: {e}")

    def delete_tf_events(self) -> None:
        # (This method is independent of CSV logging and can remain the same)
        try:
            log_dir = self._directory
            if log_dir and os.path.isdir(log_dir):
                for file in os.listdir(log_dir):
                    if "events.out.tfevents" in file:
                        os.remove(os.path.join(log_dir, file))
        except Exception as e:
             print(f"Error accessing log directory for deleting TF events: {e}")