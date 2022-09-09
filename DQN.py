import numpy as np
import torch
import torch.nn as nn
import copy

from replay_buffer import ReplayBuffer
from env import ENV
from torch.utils.tensorboard import SummaryWriter

class Network(nn.Module):

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 16)
        self.fc1.weight.data.normal_(0, 0.3)
        self.fc1.bias.data.normal_(0.1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 32)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(32, 64)
        self.fc3.weight.data.normal_(0, 0.3)
        self.fc3.bias.data.normal_(0.1)
        self.out = nn.Linear(64, n_actions)
        self.out.weight.data.normal_(0, 0.3)
        self.out.bias.data.normal_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return self.out(x)

class Double_DQN:
    def __init__(self,
                 env,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=5,
                 e_greedy_increment=0.001,
                 epoch=100
                 ):
        # print("fea:", n_features)
        self.UEs = env.UEs
        self.n_actions = env.n_actions
        self.n_features = env.n_features
        self.actions = env.actions
        self.k = env.k
        self.learning_rate = learning_rate
        self.gama = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epoch = epoch

        self.epsilon = 0
        self.learn_step_counter = 0

        # 初始化replay
        self.memory = ReplayBuffer(self.memory_size)

        self.cost_his = []

        self.eval_net = [None for _ in range(self.UEs)]
        self.target_net = [None for _ in range(self.UEs)]
        self.optimizer = [None for _ in range(self.UEs)]

        for i in range(self.UEs):

            self.eval_net[i], self.target_net[i] = Network(self.n_features , self.n_actions), Network(self.n_features,
                                                                                           self.n_actions)
            self.optimizer[i] = torch.optim.Adam(self.eval_net[i].parameters(), lr=learning_rate)

        self.loss_fun = nn.MSELoss()

    def store_memory(self, s, a, r, s_):
        self.memory.add(s, a, r, s_)

    def choose_action(self, observation):
        a = []
        for i in range(self.UEs):
            # observation = np.array(observation[i]).reshape(1, self.n_features)
            obs = np.array(observation[i]).reshape(1, self.n_features)
            obs = torch.FloatTensor(obs[:])   # 增加一个维度 i.e[1,2,3,4,5]变成[[1,2,3,4,5]]
            if np.random.uniform() < self.epsilon:
                # 选择q值最大的动作
                actions_value = self.eval_net[i](obs)
                index = torch.max(actions_value, 1)[1].data.numpy()
                index = index[0]
                action = self.actions[index]
            else:
                index = np.random.randint(0, self.n_actions)
                action = self.actions[index]
            a.append(action)
        return a

    def learn(self, step, write):
        if self.learn_step_counter % self.replace_target_iter == 0:
            for i in range(self.UEs):
                self.target_net[i].load_state_dict(self.eval_net[i].state_dict())  # 直接赋值更新权重
        self.learn_step_counter += 1

        for agent_idx, (agent_eval, agent_target, opt) in \
            enumerate(zip(self.eval_net, self.target_net, self.optimizer)):
            # 随机抽样
            obs, action, reward, obs_ = self.memory.sample(self.batch_size, agent_idx)
            actions_index = []

            rew = torch.tensor(reward, dtype=torch.float)
            action_cur = torch.from_numpy(action).to(torch.float)
            for i in range(self.batch_size):
                for j in range(self.UEs):
                    a = action_cur[i][j]
                    action_index = a[0] * self.k + a[1] / (1 / (self.k - 1))
                    actions_index.append(int(action_index))
            actions_index = torch.tensor(actions_index).reshape(self.batch_size, self.UEs, 1)
            obs_n = torch.from_numpy(obs).to(torch.float)
            obs_n_ = torch.from_numpy(obs_).to(torch.float)
            obs_n = obs_n.reshape(self.batch_size, self.UEs, self.n_features)
            obs_n_ = obs_n_.reshape(self.batch_size, self.UEs, self.n_features)

            q_target = torch.zeros((self.batch_size, self.UEs, 1))
            q_eval = agent_eval(obs_n)
            q = q_eval

            q_eval = agent_eval(obs_n).gather(-1, actions_index)
            # q_eval = torch.gather(q_eval, dim=1, index=torch.unsqueeze(action_cur, 1))
            q_next = agent_target(obs_n_).detach()

            for i in range(obs_n.shape[0]):
                for j in range(self.UEs):
                    action = torch.argmax(q[i][j], 0).detach()
                    q_target[i][j] = rew[i][j] + self.gama * q_next[i, j, action]

            loss = self.loss_fun(q_eval, q_target)
            write.add_scalar("Loss/DQN", loss, step)
            self.cost_his.append(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
#
# write = SummaryWriter(log_dir="logs")
# #
# env = ENV(3, 3, 11, 1)
# # DQN = Double_DQN(env, env.n_actions, env.n_features*5)
# DQN = Double_DQN(env)
# epoch_reward = [0.0]
# epoch_average_reward = []
# for epoch in range(1000):
#     observation = env.reset()
#     epoch_average_reward.append(epoch_reward[-1]/ (env.UEs * 100))
#     epoch_reward.append(0)
#     print("epoch:{}, cost:{}".format(epoch, epoch_average_reward[epoch]))
#     # print("reset")
#     for step in range(100):
#         o1 = copy.deepcopy(observation)
#         o2 = copy.deepcopy(observation)
#
#         action = DQN.choose_action(o1)
#         o_, reward = env.step(o2, action, is_prob=False, is_compared=False)
#         epoch_reward[-1] += np.sum(reward)
#         DQN.store_memory(o2, action, reward, o_)
#         DQN.learn(epoch, write)
#         observation = o_
#     # print("action:", action)
