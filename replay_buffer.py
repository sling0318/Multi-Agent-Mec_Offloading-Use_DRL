import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = []
        self.maxsize = int(size)
        self.next_idx = 0

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage = []
        self.next_idx = 0

    def add(self, o, a, r, o_):
        data = (o, a, r, o_)

        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.maxsize

    # 提取每个agent在replay buffter中的值
    def encode_sample(self, idxes, agent_dix):
        observations, actions, rewards, observations_ = [], [], [], []
        for i in idxes:
            data = self.storage[i]
            obs, act, rew, obs_ = data
            observations.append(np.concatenate(obs[:]))
            actions.append(act)
            rewards.append(rew[agent_dix])
            observations_.append(np.concatenate(obs_[:]))
        return np.array(observations), np.array(actions), np.array(rewards), np.array(observations_)

    # 随机抽样
    def make_index(self, batch_size):
        return [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]

    def sample(self, batch_size, agent_dix):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self.storage))
        return self.encode_sample(idxes, agent_dix)

# buffter = ReplayBuffter(100)
# o = []
# obs = np.array([238,  212,  228,  202,  213,  168,  180,  171,  195, 191,  311,  306,  408,  384,  351])
# a = np.array([238,  212,  228,  202,  213,  175,  159,  165,  189, 189,  381,  416,  443,  488,  317])
# o.append(obs)
# o.append(a)
# a = []
# action1 = np.array([1, 0.1])
# action2 = np.array([2, 0.2])
# a.append(action1)
# a.append(action2)
# r = [1000, 2000]
# o_ = []
# obs_1 = np.array([38,  212,  228,  202,  213,  175,  159,  165,  189, 189,  381,  416,  443,  488,  317])
# obs_2 = np.array([38,  212,  228,  202,  213,  175,  159,  165,  189, 189,  381,  416,  443,  488,  317])
# o_.append(obs_1)
# o_.append(obs_2)
# buffter.add(o, a, r, o_)
#
# buffter.sample(1, 1)