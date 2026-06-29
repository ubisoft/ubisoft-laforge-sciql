from sciql.core.data import Episode
from sciql.core.statistic import EpisodeStatistic

def get_traj2d_episode_statistics():
    statistics = [Return(), DiscountedReturn(), Length()]
    return statistics

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

    def __init__(self, discount: float = 0.99):

        self.name = 'discounted_return'
        self.discount = discount

    def __call__(self, episode: Episode):
        rewards = [episode[t]['reward'] for t in range(len(episode))]
        discounted_return = 0.0
        for t, reward in enumerate(rewards):
            discounted_return += (self.discount ** t) * reward
        return {'discounted_return': discounted_return}

class Length(EpisodeStatistic):

    def __init__(self):

        self.name = 'length'

    def __call__(self, episode: Episode):
        return {'length': len(episode)}