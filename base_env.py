import numpy as np
import math



class base(object):
    def __init__(self):
        # 频率
        self.Hz = 1
        self.kHz = 1000 * self.Hz
        self.mHz = 1000 * self.kHz
        self.GHz = 1000 * self.mHz

        # 数据大小
        self.bit = 1
        self.B = 8 * self.bit
        self.KB = 1024 * self.B
        self.MB = 1024 * self.KB

        self.task_cpu_cycle = np.random.randint(500, 1000)                # 处理一bit任务所需要的CPU频率
        self.task_size = np.random.randint(2 * 10**9, 3 * 10**9)  # 任务大小
        self.task_require_cycle = self.task_size * self.task_cpu_cycle    # 处理一个任务所需要的cpu频率

        # 处理任务的时间 = 任务所需的cpu频率/设备的计算能力

        self.UE_f = np.random.randint(1.5 * self.GHz, 2 * self.GHz)    # UE的计算能力
        self.MEC_f = np.random.randint(5 * self.GHz, 7 * self.GHz)   # MEC的计算能力

        # 能耗
        self.J = 1
        self.mJ = 1000 * 1000 * self.J

        self.tr_energy = 1 * self.J  # 传输1s的能耗
        self.w = 10**(-28)  # 能耗系数

        """
        进行简化
        设置传输速率为：14 Mbps
        传输时计算的是任务大小
        """
        self.r = 293 * self.MB

print(40 * math.log2(1 + (16 * 10)))
