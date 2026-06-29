from sciql.core.data import Episode
from sciql.core.statistic import EpisodeStatistic

class Return(EpisodeStatistic):

    def __init__(self):

        self.name = 'return'

    def __call__(self, episode: Episode):
        rewards = [episode[t]['reward'] for t in range(len(episode))]
        undiscounted_return = 0.0
        for t, reward in enumerate(rewards):
            undiscounted_return += reward
        return {'return': undiscounted_return}

class DiscountedReturn(EpisodeStatistic):

    def __init__(self, discount: float):

        self.name = 'discounted_return'
        self.discount = discount

    def __call__(self, episode: Episode):
        rewards = [episode[t]['reward'] for t in range(len(episode))]
        discounted_return = 0.0
        for t, reward in enumerate(rewards):
            discounted_return += (self.discount ** t) * reward
        return {'discounted_return': discounted_return}

class NormalizedReturn(EpisodeStatistic):

    def __init__(self, env_name: str):
        
        self.env_name = env_name
        self.name = 'normalized_return'

    def normalize(self, undiscounted_return: float):

        R_bounds = {
            'HalfCheetah-Mujoco': (-10.0, 10000.0)
        }
        R_min, R_max = R_bounds[self.env_name]

        return (undiscounted_return - R_min) /(R_max - R_min)


    def __call__(self, episode: Episode):
        rewards = [episode[t]['reward'] for t in range(len(episode))]
        undiscounted_return = 0.0
        for t, reward in enumerate(rewards):
            undiscounted_return += reward
        return {'normalized_return': self.normalize(undiscounted_return) * 100.0}

class Length(EpisodeStatistic):

    def __init__(self):

        self.name = 'length'

    def __call__(self, episode: Episode):
        return {'length': len(episode)}