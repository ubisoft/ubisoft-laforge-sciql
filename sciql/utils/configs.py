from omegaconf import DictConfig, ListConfig
from typing import Any, Dict, List, NamedTuple, Union

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