import os

def set_jax_determinism(deterministic: bool = True):
    """
    As it modifies the environment variables, this function
    must be called before the import of jax. 
    """
    if deterministic:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    else:
        os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=false'
        os.environ['TF_DETERMINISTIC_OPS'] = '0'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '0'