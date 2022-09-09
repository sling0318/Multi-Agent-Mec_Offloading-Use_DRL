import copy
import math

import numpy as np

class ENV():
    def __init__(self, UEs, MECs, k, lam):
        self.UEs = UEs
        self.MECs = MECs
        self.k = k

        q = np.full((k, 1), 0.)
        p = np.linspace(0, 1, k).reshape((k, 1))
        # 创建动作
        for i in range(MECs - 1):
            a = np.full((k, 1), float(i + 1))
            b = np.linspace(0, 1, k).reshape((k, 1))
            q = np.append(q, a, axis=0)
            p = np.append(p, b, axis=0)

        self.actions = np.hstack((q, p))
        self.n_actions = len(self.actions)
        self.n_features = 3 + MECs * 3
        self.discount = 0.01

        # 基本参数
        # 频率
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz
        self.nor = 10**(-7)
        self.nor1 = 10**19

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB


        # self.task_cpu_cycle = np.random.randint(2 * 10**9, 3* 10**9)

        self.UE_f = np.random.randint(1.5 * self.GHz * self.nor, 2 * self.GHz * self.nor)     # UE的计算能力
        self.MEC_f = np.random.randint(5 * self.GHz * self.nor, 7 * self.GHz * self.nor)  # MEC的计算能力
        # self.UE_f = 500 * self.mHz     # UE的计算能力
        # self.MEC_f = np.random.randint(5.2 * self.GHz, 24.3 * self.GHz)  # MEC的计算能力
        self.tr_energy = 1      # 传输能耗
        self.r = 40 * math.log2(1 + (16 * 10)) * self.MB * self.nor # 传输速率
        # self.r = 800 # 传输速率
        self.ew, self.lw = 10**(-26), 3 * 10**(-26)# 能耗系数
        # self.ew, self.lw = 0.3, 0.15 # 能耗系数
        self.et, self.lt = 1, 1
        self.local_core_max, self.local_core_min = 1.3 * self.UE_f, 0.7 * self.UE_f
        self.server_core_max, self.server_core_min = 1.3 * self.MEC_f, 0.7 * self.MEC_f
        self.uplink_max, self.uplink_min = 1.3 * self.r, 0.7 * self.r
        self.downlink_max, self.downlink_min = 1.3 * self.r, 0.7 * self.r
        self.lam = lam
        self.e = 1


    def reset(self):
        obs = []
        servers_cap = []
        new_cap = True
        for i in range(self.UEs):
            uplink, downlink = [], []
            # np.random.seed(np.random.randint(1, 1000))
            # task_size = np.random.randint(2 * 10**8 * self.nor, 3 * 10**8 * self.nor) #   任务大小
            task_size = np.random.randint(1.5 * self.mHz, 2 * self.mHz) #   任务大小
            # self.task_size = self.task_size * self.task_cpu_cycle                     # 处理一个任务所需要的cpu频率
            # task_cpu_cycle = np.random.randint(2 * 10**9 * self.nor, 3 * 10**9 * self.nor)
            task_cpu_cycle = np.random.randint(10**3, 10**5)
            local_comp = np.random.randint(0.9 * self.UE_f, 1.1 * self.UE_f)    # UE的计算能力
            for i in range(self.MECs):
                up = np.random.randint(0.9 * self.r, 1.1 * self.r)
                down = np.random.randint(0.9 * self.r, 1.1 * self.r)
                if new_cap:
                    cap = np.random.randint(0.9 * self.MEC_f, 1.1 * self.MEC_f)   # MEC计算能力
                    servers_cap.append(cap)
                uplink.append(up)
                downlink.append(down)
            observation = np.array([task_size, task_cpu_cycle, local_comp])
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

    def step(self, observation, actions_prob, is_prob=True, is_compared=True):
        if is_prob:
            actions = self.choose_action(actions_prob)
        else: actions = actions_prob
        new_cap = False
        obs_ = []
        rew, local, ran, mec = [], [], [], []
        dpg_times, local_times, ran_times, mec_times = [], [], [], []
        dpg_energys, local_energys, ran_energys, mec_energys = [], [], [], []
        total = []
        a, b, c, d = 0, 0, 0, 0
        for i in range(self.UEs):
            if i == self.UEs - 1: new_cap = True
            # 提取信息
            task_size, task_cpu_cycle, local_comp, servers_cap, uplink, downlink = \
                observation[i][0], observation[i][1], observation[i][2], observation[i][3:3+self.MECs], observation[i][3+self.MECs:3+self.MECs*2], observation[i][3+self.MECs*2:3+self.MECs*3]
            # wait_local, wait_server = np.random.randint(0, 2), np.random.randint(0, 3)

            action = actions[i]
            target_server, percen = int(action[0]), action[1]

            # 计算奖励
            # 本地和服务器上都有
            tr_time = (percen * task_size) / uplink[target_server] + self.discount * ( percen * task_size) / downlink[target_server]
            tr_energy = (self.tr_energy * percen * task_size) / uplink[target_server] + self.discount * (self.tr_energy * percen * task_size) / downlink[target_server]


            comp_local_time = task_cpu_cycle * (1 - percen) / (local_comp)
            comp_local_energy = self.lw * task_cpu_cycle * (1 - percen) * local_comp**2
            # comp_local_energy = task_size * (1 - percen) * local_comp


            comp_mec_time = (percen * task_cpu_cycle) / servers_cap[target_server]
            comp_mec_energy =self.ew * percen * task_cpu_cycle * servers_cap[target_server]**2
            # comp_mec_energy =percen * task_size * servers_cap[target_server]

            comp_time = max(comp_local_time, comp_mec_time)
            time_cost = (comp_time + tr_time) * self.et
            energy_cost = (tr_energy + comp_local_energy + comp_mec_energy) * self.e

            total_cost = self.lam * time_cost + (1 - self.lam) * energy_cost

            # reward = -total_cost

            # 全本地
            local_only_time = task_cpu_cycle/(local_comp) * self.et
            local_only_energy = self.lw * task_cpu_cycle * local_comp**2 * self.e
            # local_only_energy = task_size * local_comp
            local_only = self.lam * local_only_time + (1 - self.lam) * local_only_energy
            # print("task_cpu_cycle:", task_cpu_cycle)
            # print("local_comp", local_comp)
            # print("local_only_time:", local_only_time)
            # print("local_only_energy:", local_only_energy)
            # print("local_only:", local_only)

            # 全边缘
            mec_only_tr_time = task_size / uplink[target_server] + self.discount * task_size / downlink[target_server]
            mec_only_tr_energy = self.tr_energy * task_size / uplink[target_server] + self.discount * self.tr_energy * task_size / downlink[target_server]
            # print("mec_only_tr_time:", mec_only_tr_time)
            # print("mec_only_tr_energy:", mec_only_tr_energy)


            mec_only_comp_time = task_cpu_cycle / servers_cap[target_server]
            mec_only_comp_energy = self.ew * task_cpu_cycle * servers_cap[target_server]**2
            # mec_only_comp_energy = task_size * servers_cap[target_server]
            # print("mec_only_comp_time:", mec_only_comp_time)
            # print("mec_only_comp_energy:", mec_only_comp_energy)

            mec_only_time_cost = (mec_only_tr_time + mec_only_comp_time) * self.et
            mec_only_energy_cost = (mec_only_tr_energy + mec_only_comp_energy) * self.e

            mec_only = self.lam * mec_only_time_cost + (1 - self.lam) * mec_only_energy_cost
            # print("mec_only_time_cost:", mec_only_time_cost)
            # print("mec_only_energy_cost:", mec_only_energy_cost)
            # print("----------------------------:", servers_cap[target_server])


            # 随机卸载
            percen_ran = np.random.uniform()    # 随机卸载比例
            mec_ran = np.random.randint(self.MECs)  # 随机选择一个服务器进行卸载

            random_tr_time = (percen_ran * task_size) / uplink[mec_ran] + (self.discount * percen_ran * task_size) / downlink[mec_ran]
            random_tr_energy = (self.tr_energy * percen_ran * task_size) / uplink[mec_ran] + self.discount * (self.tr_energy * percen_ran * task_size) / downlink[mec_ran]

            random_comp_local_time = (1 - percen_ran) * task_cpu_cycle / local_comp
            random_comp_local_energy = self.lw * (1 - percen_ran) * task_cpu_cycle * local_comp**2
            # random_comp_local_energy = (1 - percen_ran) * task_size * local_comp

            random_comp_mec_time = percen_ran * task_cpu_cycle / servers_cap[mec_ran]
            random_comp_mec_energy = self.ew * percen_ran * task_cpu_cycle * servers_cap[mec_ran]**2
            # random_comp_mec_energy = percen_ran * task_size * servers_cap[mec_ran]

            random_comp_time = max(random_comp_local_time, random_comp_mec_time)
            random_time_cost = (random_comp_time + random_tr_time) * self.et
            random_energy_cost = (random_tr_energy + random_comp_local_energy + random_comp_mec_energy) * self.e


            random_total = self.lam * random_time_cost + (1 - self.lam) * random_energy_cost
            random_total_cost2 = random_energy_cost

            # if total_cost < random_total or total_cost < mec_only or total_cost < local_only:
            #     reward = -total_cost
            # else:
            #     print("惩罚")
            #     reward = -1999

            reward = -total_cost

            # a += total_cost
            # b += mec_only
            # c += local_only
            # d += random_total

            # 得到下一个observation
            x = np.random.uniform()
            y = 0.5
            if (x > y):
                local_comp = min(local_comp + np.random.randint(0, 0.2 * self.UE_f), self.local_core_max)
                for j in range(self.MECs):
                    cap = min(servers_cap[j] + np.random.randint(0, 0.3 * self.UE_f), self.server_core_max)
                    # MEC容量保持一致
                    if new_cap:
                        for x in range(self.UEs):
                            observation[x][2 + j] = cap
                    downlink[j] = min(downlink[j] + np.random.randint(0, 0.2 * self.r), self.downlink_max)
                    uplink[j] = min(uplink[j] + np.random.randint(0, 0.2 * self.r), self.uplink_max)
            else:
                local_comp = max(local_comp + np.random.randint(-0.2 * self.UE_f, 0), self.local_core_min)
                for j in range(self.MECs):
                    # MEC容量保持一致
                    if new_cap:
                        cap = max(servers_cap[j] + np.random.randint(0, 0.3 * self.UE_f), self.server_core_max)
                        for x in range(self.UEs):
                            observation[x][2 + j] = cap
                    downlink[j] = max(downlink[j] - np.random.randint(0, 0.2 * self.r), self.downlink_min)
                    uplink[j] = max(uplink[j] - np.random.randint(0, 0.2 * self.r), self.uplink_min)

            task_size = np.random.randint(10, 50)
            task_cpu_cycle = np.random.randint(10**3, 10**5)  # 处理任务所需要的CPU频率
            observation_ = np.array([task_size, task_cpu_cycle, local_comp])
            observation_ = np.hstack((observation_, servers_cap, uplink, downlink))
            obs_.append(observation_)

            rew.append(reward)
            local.append(local_only)
            mec.append(mec_only)
            ran.append(random_total)

            dpg_times.append(time_cost)
            local_times.append(local_only_time)
            mec_times.append(mec_only_time_cost)
            ran_times.append(random_time_cost)

            dpg_energys.append(energy_cost)
            local_energys.append(local_only_energy)
            mec_energys.append(mec_only_energy_cost)
            ran_energys.append(random_energy_cost)

            total.append(total_cost)

        # if (a - b > 10 * self.UEs) or (a - c > 10 * self.UEs) or (a - d > 10 * self.UEs):
        #     print("惩罚")
        #     # print(a ,b, c, d)
        #     for i in range(self.UEs):
        #         rew[i] = -999
        # else:
        #     pass

        if is_compared:
            return obs_, rew, local, mec, ran, dpg_times, local_times, mec_times, ran_times, dpg_energys, local_energys, mec_energys, ran_energys, total
        else:
            return obs_, rew, dpg_times, dpg_energys
            # return obs_, total

