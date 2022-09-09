# Multi-Agent-Mec_Offloading-Use_DRL
使用Drl来解决多智能体卸载问题

对比了MADDPG和DQN算法

环境参考论文：When Learning Joins Edge: Real-time Proportional Computation Offloading via Deep Reinforcement Learning CCF-C

目前对环境的处理为，任务在每一个step中都被处理完毕，在下一个step，对上行链路，下行链路进行随机的加减来改变状态。环境存在着一些问题，目前仍在改善。
