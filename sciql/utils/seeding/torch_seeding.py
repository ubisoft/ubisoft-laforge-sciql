import os
import torch

def set_torch_seed(seed: int):
    """
    Set torch seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def set_torch_determinism(deterministic: bool = True, hard_deterministic: bool = False):
    """
    Makes torch deterministic.
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hard_deterministic:
        torch.use_deterministic_algorithms(deterministic)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        