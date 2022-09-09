import torch
import torch.nn as nn

from Rl_net import actor, critic

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


class Maddpg(object):

    def get_train(self, env, obs_shape_n, action_shape_n):
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
        actors_tar = self.update_train(actors_cur, actors_target, 1.0)
        critics_tar = self.update_train(critics_cur, critics_target, 1.0)
        return actors_cur, critics_cur, actors_tar, critics_tar, optimizer_a, optimizer_c

    def update_train(self, agents_cur, agents_tar, tao):
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

    def agents_train(self, game_step, update_cnt, memory, obs_size, action_size,
                     actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, write):
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
                rew = []
                obs, action, reward, obs_ = memory.sample(batch_size, agent_idx)

                for i in range(batch_size):
                    r = reward[i]
                    ar = sum(r)/len(r)
                    rew.append(ar)
                # update critic
                # rew = torch.tensor(reward, dtype=torch.float)
                rew = torch.tensor(rew, dtype=torch.float)
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
                loss_t = 1e-3 * loss_pse + loss_a
                loss_t.backward()
                nn.utils.clip_grad_norm_(actor_c.parameters(), max_grad_norm)
                opt_a.step()

                write.add_scalar("Loss/Actor", loss_t, game_step)
                write.add_scalar("Loss/Critic", loss_c, game_step)

            # # save model
            # if update_cnt > save_model and update_cnt % save_fer == 0:
            #     time_now = time.strftime('%y%m_%d%H%M')
            #     print('=time:{} step:{}        save'.format(time_now, game_step))
            #     model_file_dir = os.path.join(save_dir, '{}_{}'.format(time_now, game_step))
            #     if not os.path.exists(model_file_dir):  # make the path
            #         os.makedirs(model_file_dir)
            #     for agent_idx, (a_c, a_t, c_c, c_t) in \
            #             enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
            #         torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
            #         torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
            #         torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
            #         torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

            # update the tar par
            actors_tar = self.update_train(actors_cur, actors_tar, tao)
            critics_tar = self.update_train(critics_cur, critics_tar, tao)
        return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar

