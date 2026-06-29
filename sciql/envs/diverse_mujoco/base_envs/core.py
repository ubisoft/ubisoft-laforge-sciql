from .ant import AntEnv
from .halfcheetah import HalfCheetahEnv

envs = {
    'Ant-Mujoco': AntEnv,
    'HalfCheetah-Mujoco': HalfCheetahEnv
}

time_limits = {
    'HalfCheetah-Mujoco': 1000
}