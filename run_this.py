import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from env import ENV
from replay_buffer import ReplayBuffer
from MADDPG import Maddpg
from DQN import Double_DQN
from D3QN import D3QN

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
EPOCH = 350
STEP = 200

write = SummaryWriter(log_dir="logs")

def train(ue=3, mec=7, k=11*3, lam=0.5):
    """step1:create the environment"""
    u = ue
    m = mec
    k = k
    lam = lam
    env = ENV(u, m, k, lam)    # UE: MEC:, k:
    maddpg = Maddpg()
    dqn = Double_DQN(env)
    d3qn = D3QN(env)


    print('=============================')
    print('=1 Env {} is right ...')
    print('=============================')

    """step2:create agent"""
    obs_shape_n = [env.n_features for i in range(env.UEs)]
    action_shape_n = [env.n_actions for i in range(env.UEs)]
    actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
        maddpg.get_train(env, obs_shape_n, action_shape_n)
    memory_dpg = ReplayBuffer(memory_size)
    # memory_dqn = ReplayBuffer(memory_size)

    print('=2 The {} agents are inited ...'.format(env.UEs))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    episode_rewards, episode_dqn, episode_d3qn, episode_local,  episode_mec, episode_ran = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] # sum of rewards for all agents
    episode_time_dpg,  episode_time_dqn, episode_time_d3qn, episode_time_local, episode_time_ran, episode_time_mec = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    episode_energy_dpg, episode_energy_dqn, episode_energy_d3qn, episode_energy_local, episode_energy_ran, episode_energy_mec = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    episode_total_cost = [0.0]
    # agent_rewards = [[0.0] for _ in range(env.UEs)]  # individual agent reward
    epoch_average_reward, epoch_average_dqn, epoch_average_d3qn, epoch_average_local, epoch_average_mec, epoch_average_ran= [], [], [], [], [], []
    epoch_average_time_reward, epoch_average_time_dqn, epoch_average_time_d3qn, epoch_average_time_local, epoch_average_time_mec, epoch_average_time_ran= [], [], [], [], [], []
    epoch_average_energy_reward, epoch_average_energy_dqn, epoch_average_energy_d3qn, epoch_average_energy_local, epoch_average_energy_mec, epoch_average_energy_ran= [], [], [], [], [], []
    epoch_average_total_cost = []

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

        for time_1 in range(STEP):

            action_prob = [agent(torch.from_numpy(observation).to(torch.float)).detach().cpu().numpy() \
                        for agent, observation in zip(actors_cur, obs)]
            action_dqn = dqn.choose_action(obs)
            action_d3qn = d3qn.choose_action(obs)

            o1 = copy.deepcopy(obs)
            o2 = copy.deepcopy(obs)
            obs_old = copy.deepcopy(obs)
            obs_, rew, local, mec, ran, time_dpg, time_local, time_mec, time_ran, energy_dpg, energy_local, energy_mec, energy_ran, total_cost = env.step(obs, action_prob)
            obs_dqn, rew_dqn, time_dqn, energy_dqn = env.step(o1, action_dqn, is_prob=False, is_compared=False)
            obs_d3qn, rew_d3qn, time_d3qn, energy_d3qn = env.step(o2, action_d3qn, is_prob=False, is_compared=False)


            # save the expeeinece
            memory_dpg.add(obs_old, np.concatenate(action_prob), rew, obs_)
            dqn.store_memory(obs_old, action_dqn, rew_dqn, obs_dqn)
            d3qn.store_memory(obs_old, action_d3qn, rew_d3qn, obs_d3qn)

            episode_rewards[-1] += np.sum(rew)
            episode_dqn[-1] += np.sum(rew_dqn)
            episode_d3qn[-1] += np.sum(rew_d3qn)
            episode_local[-1] += np.sum(local)
            episode_mec[-1] += np.sum(mec)
            episode_ran[-1] += np.sum(ran)

            episode_time_dpg[-1] += np.sum(time_dpg)
            episode_time_dqn[-1] += np.sum(time_dqn)
            episode_time_d3qn[-1] += np.sum(time_d3qn)
            episode_time_local[-1] += np.sum(time_local)
            episode_time_ran[-1] += np.sum(time_ran)
            episode_time_mec[-1] += np.sum(time_mec)

            episode_energy_dpg[-1] += np.sum(energy_dpg)
            episode_energy_dqn[-1] += np.sum(energy_dqn)
            episode_energy_d3qn[-1] += np.sum(energy_d3qn)
            episode_energy_local[-1] += np.sum(energy_local)
            episode_energy_mec[-1] += np.sum(energy_mec)
            episode_energy_ran[-1] += np.sum(energy_ran)
            episode_total_cost[-1] += np.sum(total_cost)
            # for i, rew in enumerate(rew):agent_rewards[i][-1] += rew

            # train agent
            if game_step > 1000 and game_step % 100 == 0:
                update_cnt, actors_cur, actors_tar, critic_cur, critic_tar = maddpg.agents_train(
                    game_step, update_cnt, memory_dpg, obs_size, action_size,
                    actors_cur, actors_tar, critic_cur, critic_tar, optimizers_a, optimizers_c, write)
                dqn.learn(game_step, write)
                d3qn.learn(game_step, write)

            # update obs
            game_step += 1
            obs = obs_
        epoch_average_reward.append(- episode_rewards[-1] / (env.UEs * STEP))
        epoch_average_dqn.append(- episode_dqn[-1] / (env.UEs * STEP))
        epoch_average_d3qn.append(- episode_d3qn[-1] / (env.UEs * STEP))
        epoch_average_local.append(episode_local[-1] / (env.UEs * STEP))
        epoch_average_mec.append(episode_mec[-1] / (env.UEs * STEP))
        epoch_average_ran.append(episode_ran[-1] / (env.UEs * STEP))

        epoch_average_time_reward.append(episode_time_dpg[-1] / (env.UEs * STEP))
        epoch_average_time_dqn.append(episode_time_dqn[-1] / (env.UEs * STEP))
        epoch_average_time_d3qn.append(episode_time_dqn[-1] / (env.UEs * STEP))
        epoch_average_time_local.append(episode_time_local[-1] / (env.UEs * STEP))
        epoch_average_time_mec.append(episode_time_mec[-1] / (env.UEs * STEP))
        epoch_average_time_ran.append(episode_time_ran[-1] / (env.UEs * STEP))

        epoch_average_energy_reward.append(episode_energy_dpg[-1] / (env.UEs * STEP))
        epoch_average_energy_dqn.append(episode_energy_dqn[-1] / (env.UEs * STEP))
        epoch_average_energy_d3qn.append(episode_energy_dqn[-1] / (env.UEs * STEP))
        epoch_average_energy_local.append(episode_energy_local[-1] / (env.UEs * STEP))
        epoch_average_energy_mec.append(episode_energy_mec[-1] / (env.UEs * STEP))
        epoch_average_energy_ran.append(episode_energy_ran[-1] / (env.UEs * STEP))
        epoch_average_total_cost.append(episode_total_cost[-1] / (env.UEs * STEP))

        episode_rewards.append(0)
        episode_dqn.append(0)
        episode_d3qn.append(0)
        episode_local.append(0)
        episode_mec.append(0)
        episode_ran.append(0)

        episode_time_dpg.append(0)
        episode_time_dqn.append(0)
        episode_time_d3qn.append(0)
        episode_time_local.append(0)
        episode_time_mec.append(0)
        episode_time_ran.append(0)

        episode_energy_dpg.append(0)
        episode_energy_dqn.append(0)
        episode_energy_d3qn.append(0)
        episode_energy_local.append(0)
        episode_energy_mec.append(0)
        episode_energy_ran.append(0)

        episode_total_cost.append(0)
        # for a_r in agent_rewards:
        #     a_r.append(0)
        # print("------reset-------")
        write.add_scalars("cost", {'MADDPG': epoch_average_total_cost[epoch],
                                   'DQN': epoch_average_dqn[epoch],
                                   'D3QN': epoch_average_d3qn[epoch],
                                   'Local': epoch_average_local[epoch],
                                   'Mec': epoch_average_mec[epoch],
                                   'random': epoch_average_ran[epoch]}, epoch)
        # write.add_scalars("cost", {'MADDPG': - episode_rewards[-1] /STEP,
        #                            # 'DQN': epoch_average_dqn[epoch],
        #                            'Local': episode_local[-1] / STEP,
        #                            'Mec': episode_mec[-1] / STEP,
        #                            'random': episode_ran[-1] / STEP}, epoch)
        write.add_scalars("cost/energy", {'MADDPG': epoch_average_energy_reward[epoch],
                                     'DQN': epoch_average_energy_dqn[epoch],
                                     'D3QN': epoch_average_energy_d3qn[epoch],
                                     'Local': epoch_average_energy_local[epoch],
                                     'Mec': epoch_average_energy_mec[epoch],
                                     'random': epoch_average_energy_ran[epoch]}, epoch)
        write.add_scalars("cost/delay", {'MADDPG': epoch_average_time_reward[epoch],
                                    'DQN': epoch_average_time_dqn[epoch],
                                    'D3QN': epoch_average_time_d3qn[epoch],
                                    'Local': epoch_average_time_local[epoch],
                                    'Mec': epoch_average_time_mec[epoch],
                                    'random': epoch_average_time_ran[epoch]}, epoch)
        # print("epoch:{},MADDPG:{}".format(epoch, epoch_average_total_cost[epoch]))
        # # print("epoch:{},DQN:{}".format(epoch, epoch_average_dqn[epoch]))
        # print("epoch:{},Local:{}".format(epoch, epoch_average_local[epoch]))
        # print("epoch:{},Mec:{}".format(epoch, epoch_average_mec[epoch]))
        # print("epoch:{},random:{}".format(epoch, epoch_average_ran[epoch]))
        # if epoch_average_mec[epoch] > epoch_average_reward[epoch]:
        #     print("True")
        # print("---------------------------------------")
    # return a



if __name__ == '__main__':
    # for i in range(5):
    #     cost = train(i + 10)
    #     print(i + 10, "cost:", cost)
    #     write.add_scalar("cost", cost, i + 10)
    #     write.close()
    train()
