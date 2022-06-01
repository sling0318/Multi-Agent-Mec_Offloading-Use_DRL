import copy

import numpy as np

class ENV():
    def __init__(self, UEs, MECs, k):
        self.UEs = UEs
        self.MECs = MECs
        self.k = k

        q = np.full((k, 1), 0.)
        p = np.linspace(0, 1, k).reshape((k, 1))
        for i in range(MECs-1):
            a = np.full((k, 1), float(i + 1))
            b = np.linspace(0, 1, k).reshape((k, 1))
            q = np.append(q, a, axis=0)
            p = np.append(p, b, axis=0)

        self.actions = np.hstack((q, p))
        self.n_actions = len(self.actions)
        self.n_features = 2 + MECs * 3
        self.beta_local, self.cycle_perbyte, self.beta_tr = 0.4, 1, 10,   # 本地能耗， CPU周期， 传输能耗
        self.beta_re, self.energy_per_r, self.discount = 0.1, 0.3, 0.01
        self.local_core_max, self.local_core_min = 200, 50
        self.server_core_max, self.server_core_min = 400, 150
        self.uplink_max, self.uplink_min = 350, 100
        self.downlink_max, self.downlink_min = 600, 250


    def reset(self):
        obs = []
        servers_cap = []
        new_cap = True
        for i in range(self.UEs):
            uplink, downlink = [], []
            np.random.seed(np.random.randint(1, 1000))
            workload = np.random.randint(2000, 3000)  # 定义工作来量
            local_comp = np.random.randint(90, 110)  # 本地计算资源
            for i in range(self.MECs):
                up = np.random.randint(150, 200)
                down = np.random.randint(300, 500)
                if new_cap:
                    cap = np.random.randint(300, 400)   # MEC计算能力
                    servers_cap.append(cap)
                uplink.append(up)
                downlink.append(down)
            observation = np.array([workload, local_comp])
            observation = np.hstack((observation, servers_cap, uplink, downlink))
            obs.append(observation)
            new_cap = False
        return obs

    def choose_action(self, prob):
        """
        根据概率选择动作
        :param env:
        :param prob:
        :return: [[target_server, percentage]]
        """
        action_choice = np.linspace(0, 1, self.k)
        actions = []
        for i in range(self.UEs):
            a = np.random.choice(a=(self.MECs * self.k), p=prob[i])  # 在数组p中从a个数字中以概率p选中一个
            target_server = int(a / self.k)
            percen = action_choice[a % self.k]
            action = [target_server, percen]
            actions.append(action)
        return actions

    def step(self, observation, actions_prob, time1, is_prob=True, is_compared=True):
        if is_prob:
            actions = self.choose_action(actions_prob)
        else: actions = actions_prob
        new_cap = False
        obs_ = []
        rew, local, ran, mec = [], [], [], []
        dpg_times, local_times, ran_times, mec_times = [], [], [], []
        dpg_energys, local_energys, ran_energys, mec_energys = [], [], [], []
        for i in range(self.UEs):
            if i == self.UEs - 1: new_cap = True
            # 提取信息
            workload, local_comp, servers_cap, uplink, downlink = \
                observation[i][0], observation[i][1], observation[i][2:2+self.MECs], observation[i][2+self.MECs:2+self.MECs*2], observation[i][2+self.MECs*2:2+self.MECs*3]
            wait_local, wait_server = np.random.randint(0, 2), np.random.randint(0, 3)
            action = actions[i]
            # print("action:", action)
            target_server, percen = int(action[0]), action[1]

            # 计算奖励
            # 本地和服务器上都有
            local_time = workload * self.cycle_perbyte * (1 - percen) / (local_comp) + wait_local

            local_energy = self.beta_local * workload * (1 - percen)


            remote_time = workload * self.cycle_perbyte * percen / (servers_cap[target_server]) + wait_server + \
                          workload * percen / (uplink[target_server]) + self.discount * workload / (downlink[target_server])

            remote_energy = self.beta_re * workload * percen + self.beta_tr * workload * percen/(uplink[target_server]) + self.beta_tr * self.discount * workload / (downlink[target_server])

            time_cost = max(local_time, remote_time)

            energy_cost = local_energy + remote_energy

            total_cost = 0.6 * time_cost + 0.4 * energy_cost
            total_cost2 = energy_cost

            reward = -total_cost

            # 全本地
            local_only_time = workload * self.cycle_perbyte /(local_comp)

            local_only_energy = self.beta_local * workload

            local_only = 0.6 * local_only_time + 0.4 * local_only_energy

            # 全边缘
            remote_only_time = workload * self.cycle_perbyte / (servers_cap[target_server]) + wait_server + \
                          workload / (uplink[target_server]) + self.discount * workload / (downlink[target_server])

            remote_only_energy = self.beta_re * workload + self.beta_tr * workload / (uplink[target_server]) + self.beta_tr * self.discount * workload / (downlink[target_server])

            remote_only = 0.6 * remote_only_time + 0.4 * remote_only_energy

            # 随机卸载
            percen_ran = np.random.uniform()
            mec_ran = np.random.randint(self.MECs)
            random_local_time = workload * self.cycle_perbyte * (1 - percen_ran) / (local_comp) + wait_local

            random_local_energy = self.beta_local * workload * (1 - percen_ran)

            random_remote_time = workload * self.cycle_perbyte * percen_ran / (servers_cap[mec_ran]) + wait_server + \
                          workload * percen_ran / (uplink[mec_ran]) + self.discount * workload / (
                          downlink[mec_ran])

            random_remote_energy = self.beta_re * workload * percen_ran + self.beta_tr * workload * percen_ran/(uplink[target_server]) + self.beta_tr * self.discount * workload / (downlink[target_server])

            random_time_cost = max(random_local_time, random_remote_time)

            random_energy_cost = random_local_energy + random_remote_energy

            random_total_cost = 0.6 * random_time_cost + 0.4 * random_energy_cost
            random_total_cost2 = random_energy_cost


            # 得到下一个observation
            a = np.random.uniform()
            b = 0.5
            if (a > b):
                local_comp = min(local_comp + np.random.randint(0, 6), self.local_core_max)
                for j in range(self.MECs):
                    cap = min(servers_cap[j] + np.random.randint(0, 15), self.server_core_max)
                    # MEC容量保持一致
                    if new_cap:
                        for x in range(self.UEs):
                            observation[x][2 + j] = cap
                    downlink[j] = min(downlink[j] + np.random.randint(0, 8), self.downlink_max)
                    uplink[j] = min(uplink[j] + np.random.randint(0, 5), self.uplink_max)
            else:
                local_comp = max(local_comp + np.random.randint(-5, 0), self.local_core_min)
                for j in range(self.MECs):
                    # MEC容量保持一致
                    if new_cap:
                        cap = max(servers_cap[j] + np.random.randint(0, 15), self.server_core_max)
                        for x in range(self.UEs):
                            observation[x][2 + j] = cap
                    downlink[j] = max(downlink[j] - np.random.randint(0, 8), self.downlink_min)
                    uplink[j] = max(uplink[j] - np.random.randint(0, 5), self.uplink_min)
            # for x in range(self.UEs):
            #     self.observation[x][2+target_server] = self.observation[x][2+target_server] - workload * percen * discount
            workload += np.random.randint(-100, 200)
            observation_ = np.array([workload, local_comp])
            observation_ = np.hstack((observation_, servers_cap, uplink, downlink))
            obs_.append(observation_)
            rew.append(reward)
            local.append(local_only)
            ran.append(random_total_cost)
            mec.append(remote_only)
            dpg_times.append(time_cost)
            local_times.append(local_only_time)
            ran_times.append(random_time_cost)
            mec_times.append(remote_only_time)
            dpg_energys.append(energy_cost)
            local_energys.append(local_only_energy)
            ran_energys.append(random_energy_cost)
            mec_energys.append(remote_only_energy)

        if is_compared:
            return obs_, rew, local, mec, ran, dpg_times, local_times, mec_times, ran_times, dpg_energys, local_energys, mec_energys, ran_energys
        else:
            return obs_, rew

