import copy
import time
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from env import ENV
from Rl_net import actor, critic
from replay_buffer import ReplayBuffer

learning_start_step = 200
learning_fre = 5
batch_size = 64
gamma = 0.9
lr = 0.01
max_grad_norm = 0.5
save_model = 40
save_dir = "models/simple_adversary"
save_fer = 400
tao = 0.01
memory_size = 2000
EPOCH = 200


def get_train(env, obs_shape_n, action_shape_n):
    actors_cur = [None for _ in range(env.UEs)]
    critics_cur = [None for _ in range(env.UEs)]
    actors_target = [None for _ in range(env.UEs)]
    critics_target = [None for _ in range(env.UEs)]
    optimizer_a = [None for _ in range(env.UEs)]
    optimizer_c = [None for _ in range(env.UEs)]


    for i in range(env.UEs):
        actors_cur[i] = actor(obs_shape_n[i], action_shape_n[i])
        critics_cur[i] = critic(sum(obs_shape_n), sum(action_shape_n))
        actors_target[i] = actor(obs_shape_n[i], action_shape_n[i])
        critics_target[i] = critic(sum(obs_shape_n), sum(action_shape_n))
        optimizer_a[i] = torch.optim.Adam(actors_cur[i].parameters(), lr=lr)
        optimizer_c[i] = torch.optim.Adam(critics_cur[i].parameters(), lr=lr)
    actors_tar = update_train(actors_cur, actors_target, 1.0)
    critics_tar = update_train(critics_cur, critics_target, 1.0)
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizer_a, optimizer_c

def update_train(agents_cur, agents_tar, tao):
    """
    用于更新target网络，
    这个方法不同于直接复制，但结果一样
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def agents_train(game_step, update_cnt, memory, obs_size, action_size,
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """
    par:
    |input: the data for training
    |output: the data for next update
    """

    # 训练
    if (game_step > learning_start_step) and (game_step % learning_fre == 0):
        if update_cnt == 0: print('\r=start training...' + ''*100)
        update_cnt += 1

        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue

            # 随机抽样
            obs, action, reward, obs_ = memory.sample(batch_size, agent_idx)

            # update critic
            rew = torch.tensor(reward, dtype=torch.float)
            action_cur = torch.from_numpy(action).to(torch.float)
            obs_n = torch.from_numpy(obs).to(torch.float)
            obs_n_ = torch.from_numpy(obs_).to(torch.float)
            action_tar = torch.cat([a_t(obs_n_[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n, action_cur).reshape(-1)     # q
            q_ = critic_t(obs_n_, action_tar).reshape(-1)   # q_
            tar_value = q_ * gamma + rew
            loss_c = torch.nn.MSELoss()(q, tar_value)
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), max_grad_norm)
            opt_c.step()

            # update Actor
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c(
                obs_n_[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            action_cur[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n, action_cur)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
            opt_a.step()

        # save model
        if update_cnt > save_model and update_cnt % save_fer == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(save_dir, '{}_{}'.format(time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.makedirs(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_train(actors_cur, actors_tar, tao)
        critics_tar = update_train(critics_cur, critics_tar, tao)
    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def choose_action(env, prob):
    """
    根据概率选择动作
    :param env:
    :param prob:
    :return: [[target_server, percentage]]
    """
    action_choice = np.linspace(0, 1, env.k)
    actions = []
    for i in range(env.UEs):
        a = np.random.choice(a=(env.MECs * env.k), p=prob[i])  # 在数组p中从a个数字中以概率p选中一个
        target_server = int(a / env.k)
        percen = action_choice[a % env.k]
        action = [target_server, percen]
        actions.append(action)
    return actions

def train():
    write = SummaryWriter(log_dir="logs")

    """step1:create the environment"""
    env = ENV(8, 7, 256)    # UE: MEC:, k:

    print('=============================')
    print('=1 Env {} is right ...')
    print('=============================')

    """step2:create agent"""
    obs_shape_n = [env.n_features for i in range(env.UEs)]
    action_shape_n = [env.n_actions for i in range(env.UEs)]
    actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
        get_train(env, obs_shape_n, action_shape_n)
    memory = ReplayBuffer(memory_size)

    print('=2 The {} agents are inited ...'.format(env.UEs))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    episode_rewards, episode_local,  episode_mec, episode_ran = [0.0], [0.0], [0.0], [0.0]  # sum of rewards for all agents
    episode_time_dpg,  episode_time_local, episode_time_ran, episode_time_mec = [0.0], [0.0], [0.0], [0.0]
    episode_energy_dpg,  episode_energy_local, episode_energy_ran, episode_energy_mec = [0.0], [0.0], [0.0], [0.0]
    agent_rewards = [[0.0] for _ in range(env.UEs)]  # individual agent reward
    epoch_average_reward, epoch_average_local, epoch_average_mec, epoch_average_ran= [], [], [], []
    epoch_average_time_reward, epoch_average_time_local, epoch_average_time_mec, epoch_average_time_ran= [], [], [], []
    epoch_average_energy_reward, epoch_average_energy_local, epoch_average_energy_mec, epoch_average_energy_ran= [], [], [], []

    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')

    for epoch in range(EPOCH):
        obs = env.reset()
        epoch_average_reward.append(- episode_rewards[-1] / (env.UEs * 108))
        epoch_average_local.append( episode_local[-1] / (env.UEs * 108))
        epoch_average_mec.append( episode_mec[-1] / (env.UEs * 108))
        epoch_average_ran.append( episode_ran[-1] / (env.UEs * 108))
        epoch_average_time_reward.append( episode_time_dpg[-1] / (env.UEs * 108))
        epoch_average_time_local.append( episode_time_local[-1] / (env.UEs * 108))
        epoch_average_time_mec.append( episode_time_mec[-1] / (env.UEs * 108))
        epoch_average_time_ran.append( episode_time_ran[-1] / (env.UEs * 108))
        epoch_average_energy_reward.append( episode_energy_dpg[-1] / (env.UEs * 108))
        epoch_average_energy_local.append( episode_energy_local[-1] / (env.UEs * 108))
        epoch_average_energy_mec.append( episode_energy_mec[-1] / (env.UEs * 108))
        epoch_average_energy_ran.append( episode_energy_ran[-1] / (env.UEs * 108))
        episode_rewards.append(0)
        episode_local.append(0)
        episode_mec.append(0)
        episode_ran.append(0)
        episode_time_dpg.append(0)
        episode_time_local.append(0)
        episode_time_mec.append(0)
        episode_time_ran.append(0)
        episode_energy_dpg.append(0)
        episode_energy_local.append(0)
        episode_energy_mec.append(0)
        episode_energy_ran.append(0)
        for a_r in agent_rewards:
            a_r.append(0)
        # print("------reset-------")
        write.add_scalars("cost", {'MADDPG': epoch_average_reward[epoch],
                                   'Local': epoch_average_local[epoch],
                                   'Mec': epoch_average_mec[epoch],
                                   'random': epoch_average_ran[epoch]}, epoch)
        write.add_scalars("delay", {'MADDPG': epoch_average_time_reward[epoch],
                                   'Local': epoch_average_time_local[epoch],
                                   'Mec': epoch_average_time_mec[epoch],
                                   'random': epoch_average_time_ran[epoch]}, epoch)




        for time_1 in range(108):
            r = 0
            action_prob = [agent(torch.from_numpy(observation).to(torch.float)).detach().cpu().numpy() \
                        for agent, observation in zip(actors_cur, obs)]
            # action_n = choose_action(env, action_prob)
            obs_old = copy.deepcopy(obs)
            obs_, rew, local, mec, ran, time_dpg, time_local, time_mec, time_ran, energy_dpg, energy_local, energy_mec, energy_ran = env.step(obs, action_prob, time_1)


            # save the expeeinece
            memory.add(obs_old, np.concatenate(action_prob), rew, obs_)
            episode_rewards[-1] += np.sum(rew)
            episode_local[-1] += np.sum(local)
            episode_mec[-1] += np.sum(mec)
            episode_ran[-1] += np.sum(ran)
            episode_time_dpg[-1] += np.sum(time_dpg)
            episode_time_local[-1] += np.sum(time_local)
            episode_time_ran[-1] += np.sum(time_ran)
            episode_time_mec[-1] += np.sum(time_mec)
            episode_energy_dpg[-1] += np.sum(energy_dpg)
            episode_energy_local[-1] += np.sum(energy_local)
            episode_energy_mec[-1] += np.sum(energy_mec)
            episode_energy_ran[-1] += np.sum(energy_ran)
            for i, rew in enumerate(rew):agent_rewards[i][-1] += rew

            # train agent
            update_cnt, actors_cur, actors_tar, critic_cur, critic_tar = agents_train(
                game_step, update_cnt, memory, obs_size, action_size,
                actors_cur, actors_tar, critic_cur, critic_tar, optimizers_a, optimizers_c)

            # update obs
            game_step += 1
            obs = obs_
    write.close()



if __name__ == '__main__':
    train()

import copy
import time
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from env import ENV
from Rl_net import actor, critic
from replay_buffer import ReplayBuffer

learning_start_step = 200
learning_fre = 5
batch_size = 64
gamma = 0.9
lr = 0.01
max_grad_norm = 0.5
save_model = 40
save_dir = "models/simple_adversary"
save_fer = 400
tao = 0.01
memory_size = 2000
EPOCH = 200


def get_train(env, obs_shape_n, action_shape_n):
    actors_cur = [None for _ in range(env.UEs)]
    critics_cur = [None for _ in range(env.UEs)]
    actors_target = [None for _ in range(env.UEs)]
    critics_target = [None for _ in range(env.UEs)]
    optimizer_a = [None for _ in range(env.UEs)]
    optimizer_c = [None for _ in range(env.UEs)]


    for i in range(env.UEs):
        actors_cur[i] = actor(obs_shape_n[i], action_shape_n[i])
        critics_cur[i] = critic(sum(obs_shape_n), sum(action_shape_n))
        actors_target[i] = actor(obs_shape_n[i], action_shape_n[i])
        critics_target[i] = critic(sum(obs_shape_n), sum(action_shape_n))
        optimizer_a[i] = torch.optim.Adam(actors_cur[i].parameters(), lr=lr)
        optimizer_c[i] = torch.optim.Adam(critics_cur[i].parameters(), lr=lr)
    actors_tar = update_train(actors_cur, actors_target, 1.0)
    critics_tar = update_train(critics_cur, critics_target, 1.0)
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizer_a, optimizer_c

def update_train(agents_cur, agents_tar, tao):
    """
    用于更新target网络，
    这个方法不同于直接复制，但结果一样
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def agents_train(game_step, update_cnt, memory, obs_size, action_size,
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """
    par:
    |input: the data for training
    |output: the data for next update
    """

    # 训练
    if (game_step > learning_start_step) and (game_step % learning_fre == 0):
        if update_cnt == 0: print('\r=start training...' + ''*100)
        update_cnt += 1

        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue

            # 随机抽样
            obs, action, reward, obs_ = memory.sample(batch_size, agent_idx)

            # update critic
            rew = torch.tensor(reward, dtype=torch.float)
            action_cur = torch.from_numpy(action).to(torch.float)
            obs_n = torch.from_numpy(obs).to(torch.float)
            obs_n_ = torch.from_numpy(obs_).to(torch.float)
            action_tar = torch.cat([a_t(obs_n_[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n, action_cur).reshape(-1)     # q
            q_ = critic_t(obs_n_, action_tar).reshape(-1)   # q_
            tar_value = q_ * gamma + rew
            loss_c = torch.nn.MSELoss()(q, tar_value)
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), max_grad_norm)
            opt_c.step()

            # update Actor
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c(
                obs_n_[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            action_cur[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n, action_cur)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
            opt_a.step()

        # save model
        if update_cnt > save_model and update_cnt % save_fer == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(save_dir, '{}_{}'.format(time_now, game_step))
            if not os.path.exists(model_file_dir):  # make the path
                os.makedirs(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                    enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_train(actors_cur, actors_tar, tao)
        critics_tar = update_train(critics_cur, critics_tar, tao)
    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def choose_action(env, prob):
    """
    根据概率选择动作
    :param env:
    :param prob:
    :return: [[target_server, percentage]]
    """
    action_choice = np.linspace(0, 1, env.k)
    actions = []
    for i in range(env.UEs):
        a = np.random.choice(a=(env.MECs * env.k), p=prob[i])  # 在数组p中从a个数字中以概率p选中一个
        target_server = int(a / env.k)
        percen = action_choice[a % env.k]
        action = [target_server, percen]
        actions.append(action)
    return actions

def train():
    write = SummaryWriter(log_dir="logs")

    """step1:create the environment"""
    env = ENV(8, 7, 256)    # UE: MEC:, k:

    print('=============================')
    print('=1 Env {} is right ...')
    print('=============================')

    """step2:create agent"""
    obs_shape_n = [env.n_features for i in range(env.UEs)]
    action_shape_n = [env.n_actions for i in range(env.UEs)]
    actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
        get_train(env, obs_shape_n, action_shape_n)
    memory = ReplayBuffer(memory_size)

    print('=2 The {} agents are inited ...'.format(env.UEs))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    episode_rewards, episode_local,  episode_mec, episode_ran = [0.0], [0.0], [0.0], [0.0]  # sum of rewards for all agents
    episode_time_dpg,  episode_time_local, episode_time_ran, episode_time_mec = [0.0], [0.0], [0.0], [0.0]
    episode_energy_dpg,  episode_energy_local, episode_energy_ran, episode_energy_mec = [0.0], [0.0], [0.0], [0.0]
    agent_rewards = [[0.0] for _ in range(env.UEs)]  # individual agent reward
    epoch_average_reward, epoch_average_local, epoch_average_mec, epoch_average_ran= [], [], [], []
    epoch_average_time_reward, epoch_average_time_local, epoch_average_time_mec, epoch_average_time_ran= [], [], [], []
    epoch_average_energy_reward, epoch_average_energy_local, epoch_average_energy_mec, epoch_average_energy_ran= [], [], [], []

    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')

    for epoch in range(EPOCH):
        obs = env.reset()
        epoch_average_reward.append(- episode_rewards[-1] / (env.UEs * 108))
        epoch_average_local.append( episode_local[-1] / (env.UEs * 108))
        epoch_average_mec.append( episode_mec[-1] / (env.UEs * 108))
        epoch_average_ran.append( episode_ran[-1] / (env.UEs * 108))
        epoch_average_time_reward.append( episode_time_dpg[-1] / (env.UEs * 108))
        epoch_average_time_local.append( episode_time_local[-1] / (env.UEs * 108))
        epoch_average_time_mec.append( episode_time_mec[-1] / (env.UEs * 108))
        epoch_average_time_ran.append( episode_time_ran[-1] / (env.UEs * 108))
        epoch_average_energy_reward.append( episode_energy_dpg[-1] / (env.UEs * 108))
        epoch_average_energy_local.append( episode_energy_local[-1] / (env.UEs * 108))
        epoch_average_energy_mec.append( episode_energy_mec[-1] / (env.UEs * 108))
        epoch_average_energy_ran.append( episode_energy_ran[-1] / (env.UEs * 108))
        episode_rewards.append(0)
        episode_local.append(0)
        episode_mec.append(0)
        episode_ran.append(0)
        episode_time_dpg.append(0)
        episode_time_local.append(0)
        episode_time_mec.append(0)
        episode_time_ran.append(0)
        episode_energy_dpg.append(0)
        episode_energy_local.append(0)
        episode_energy_mec.append(0)
        episode_energy_ran.append(0)
        for a_r in agent_rewards:
            a_r.append(0)
        # print("------reset-------")
        write.add_scalars("cost", {'MADDPG': epoch_average_reward[epoch],
                                   'Local': epoch_average_local[epoch],
                                   'Mec': epoch_average_mec[epoch],
                                   'random': epoch_average_ran[epoch]}, epoch)
        write.add_scalars("delay", {'MADDPG': epoch_average_time_reward[epoch],
                                   'Local': epoch_average_time_local[epoch],
                                   'Mec': epoch_average_time_mec[epoch],
                                   'random': epoch_average_time_ran[epoch]}, epoch)




        for time_1 in range(108):
            r = 0
            action_prob = [agent(torch.from_numpy(observation).to(torch.float)).detach().cpu().numpy() \
                        for agent, observation in zip(actors_cur, obs)]
            # action_n = choose_action(env, action_prob)
            obs_old = copy.deepcopy(obs)
            obs_, rew, local, mec, ran, time_dpg, time_local, time_mec, time_ran, energy_dpg, energy_local, energy_mec, energy_ran = env.step(obs, action_prob, time_1)


            # save the expeeinece
            memory.add(obs_old, np.concatenate(action_prob), rew, obs_)
            episode_rewards[-1] += np.sum(rew)
            episode_local[-1] += np.sum(local)
            episode_mec[-1] += np.sum(mec)
            episode_ran[-1] += np.sum(ran)
            episode_time_dpg[-1] += np.sum(time_dpg)
            episode_time_local[-1] += np.sum(time_local)
            episode_time_ran[-1] += np.sum(time_ran)
            episode_time_mec[-1] += np.sum(time_mec)
            episode_energy_dpg[-1] += np.sum(energy_dpg)
            episode_energy_local[-1] += np.sum(energy_local)
            episode_energy_mec[-1] += np.sum(energy_mec)
            episode_energy_ran[-1] += np.sum(energy_ran)
            for i, rew in enumerate(rew):agent_rewards[i][-1] += rew

            # train agent
            update_cnt, actors_cur, actors_tar, critic_cur, critic_tar = agents_train(
                game_step, update_cnt, memory, obs_size, action_size,
                actors_cur, actors_tar, critic_cur, critic_tar, optimizers_a, optimizers_c)

            # update obs
            game_step += 1
            obs = obs_
    write.close()



if __name__ == '__main__':
    train()

