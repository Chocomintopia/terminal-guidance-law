import numpy as np
import math
import matplotlib.pyplot as plt
import time
from testEnv import simulitiveEnv
from hindsightEnv import HindsightEnv
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from copy import deepcopy
from per_utils import device, set_seed
from per_buffer import ReplayBuffer, PrioritizedReplayBuffer

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# Hyper Parameters
HINDSIGHT_SIZE = 4 # 存储了8个量用于 Hindsight地计算state (xm,ym,vm,theta_m,xt,yt,vt,theta_t)


BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.1               # greedy policy
GAMMA = 0.9                 # reward discount
ALPHA = 0.1                 # 与采样概率有关的alpha
BETA = 0.1                  # 与计算重要性权值有关的beta
ETA = 0                     # Delta更新的步长
TARGET_REPLACE_ITER = 1000  # 参数更新间隔
MEMORY_CAPACITY = 10000     # 经验池容量

REWARD = 0.1

N_ACTIONS = 25 # action的维度 N1(2,3,4,5,6) \times N2(0,0.5,1,1.5,2)
N_STATES = 6   # state的维度 (r,v_rela,q,q^,v,\theta_m)
LOSS = []


# class Net(nn.Module):
#     def __init__(self, ):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_STATES, 128)     # 输入层将状态嵌入作为输入 并产生128维输出至隐藏层
#         self.fc1.weight.data.normal_(0, 0.1)   # initialization
#         self.out = nn.Linear(128, N_ACTIONS)    # 输出层接受隐藏层的输出值并针对可能的动作值评估其Q价值 并且选择价值最高的动作作为输出
#         self.out.weight.data.normal_(0, 0.1)   # initialization


#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value # 返回动作价值 Q

class PER(object):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS)
        ).to(device())
        self.target_model = deepcopy(self.model).to(device())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.learn_step_counter = 0                                     # for target updating
        self.memory = PrioritizedReplayBuffer(N_STATES, N_ACTIONS, MEMORY_CAPACITY, EPSILON, ALPHA, BETA)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR) # 利用 Adam 进行参数优化

        #parameters for training
        self.batch_num = 0
        self.batch_count = 0
        self.epi_len = 0
        self.epi_state = None       # s
        self.epi_actions = None     # a
        self.epi_rewards = None     # r
        self.epi_state_ = None      # s'

    def update(self, batch, weights=None):
        
        state, action, reward, next_state, done = batch

        b_s = torch.FloatTensor(state) # s
        b_a = torch.FloatTensor(action) # a
        b_r = torch.FloatTensor(reward) # r
        b_s_ = torch.FloatTensor(next_state) # s'
        b_done = torch.BoolTensor(done)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.model(b_s).gather(1, b_a.to(torch.long))[:, 0]
        next_q_value = self.target_model(b_s_).max(1)[0].detach()
        target = (b_r + GAMMA * next_q_value * (1 - b_done.to(torch.float)) )

        td_error = torch.abs(curr_q_value - target).detach()

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), td_error

# #################
#         Q_next = self.target_model(b_s_).max(dim=1).detach()
#         Q_target = b_r + GAMMA * (1 - b_done.to(torch.float)) * Q_next
#         Q = self.model(b_s)[torch.arange(len(b_a)), b_a.to(torch.long).flatten()]

#         assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

#         if weights is None:
#             weights = torch.ones_like(Q)

#         td_error = torch.abs(Q - Q_target).detach()
#         loss = torch.mean((Q - Q_target)**2 * weights)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return loss.item(), td_error

    def learn(self): # 细节待改 2023/5/18 12:33
        # 间歇性更新目标网络参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: 
            self.target_model.load_state_dict(self.model.state_dict())
        self.learn_step_counter += 1

        # 随机小批量从经验池中采样
        batch, weights, tree_idxs = self.memory.sample(BATCH_SIZE)
        
        # 计算TD误差
        loss, td_error = self.model.update(batch, weights=weights)

        # 更新优先级
        self.memory.update_priorities(tree_idxs, td_error.numpy())




    def choose_action(self, x): # \epsilon-greedy选择动作
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() > EPSILON:   # greedy
            actions_value = self.model.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]   # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action

    def choose_action_greedy(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.model.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]   # return the argmax index
        return action


class Normalizer():
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

if __name__ == '__main__':

    log_dir = 'log\per1'
    logger = SummaryWriter(logdir=log_dir)

    np.random.seed(0)
    per = PER()
    # 先初始化环境，初始一个action
    xt0 = 6 * math.pow(10, 3) # 目标初始位置 x
    yt0 = 8 * math.pow(10, 3) # 目标初始位置 y
    vt0 = 300.0 # 目标初始速度
    xm0 = 0.0 # 导弹初始位置 x
    ym0 = 11000.0 # 导弹初始位置 y
    vm0 = 1000.0 # 导弹初始速度
    Initial_heading_angle_receive = -25 # 导弹初始弹道倾角
    distance_maneuver = 2 #这个代表开始运动了2m之后，就开始各个方向机动了。
    Value_direction_maneuver = np.random.randint(1, 4) # 机动方式
    # Value_direction_maneuver = 2
    Value_target_acceleration = 2 # 目标初始加速度
    arg = (xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver, Value_direction_maneuver,
           Value_target_acceleration)
    main_loop_size = 120
    # sigma = main_loop_size+1 #wty

    sigma = 0

    horizon = 700
    state_dim = 6
    action_dim = 1
    action_bound = 200
    print('\nCollecting experience...')

    trajectory_m_all = []
    trajectory_t_all = []

    count = 0


    for i_epi in range(main_loop_size):
        hit_num = 0 # 命中率



        xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver, \
        Value_direction_maneuver, Value_target_acceleration = arg
        env_simple = simulitiveEnv(xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive,
                                   distance_maneuver, Value_direction_maneuver, Value_target_acceleration, horizon)
        state = env_simple.reset()
        ep_r = 0 # episode reward
        done = False
        num_plays = 0.
        normalizer = Normalizer(state_dim)

        trajectory_m = np.zeros((horizon, HINDSIGHT_SIZE + 1)) # 比目标飞行物多存一个动作
        trajectory_t = np.zeros((horizon, HINDSIGHT_SIZE))
        action_radio_hindsight = 0.0

        hit_finally_flag = False
        interrupt_flag = False
        done_flag = 0


        for i in range(horizon):
            # #add gauss noise
            # import random
            # noise = []
            # for i in range(state.size):
            #     noise.append(random.gauss(0, 0.1))
            # state_noise = state + noise
            action_radio = per.choose_action(state)
            action = env_simple.choose_action_RL(action_radio)  # 将动作索引转变为实际的动作
            # action = env_simple.choose_action()
            state_, reward, done, hit_flag = env_simple._step(action) #执行动作，进入下一状态

            action_radio_hindsight = action_radio

            xm_now = env_simple.xm[-1]
            ym_now = env_simple.ym[-1]
            vm_now = env_simple.vm[-1]
            theta_m_now = env_simple.theta_m[-1]
            xt_now = env_simple.xt[-1]
            yt_now = env_simple.yt[-1]
            vt_now = env_simple.vt[-1]
            theta_t_now = env_simple.theta_t[-1]

            # 当相对距离开始连续变大时 则没打中 停止存储transition
            if len(env_simple.r) > 3:
                if env_simple.r[-1] > env_simple.r[-2]:
                    done_flag += 1
            if not done and done_flag >= 2:    #interrupt_flag代表着这次打偏了，没打中，也会结束
                interrupt_flag = True
                done_flag = 0
            
            if not interrupt_flag:
                per.memory.add((state, action_radio, reward, state_, done))


            # logger.add_scalar('step_reward',reward , i)

            # ep_r += reward
            if per.memory.count == 0:

                # 间歇性更新目标网络参数
                if per.learn_step_counter % TARGET_REPLACE_ITER == 0: 
                    per.target_model.load_state_dict(per.model.state_dict())
                per.learn_step_counter += 1

                # 随机小批量从经验池中采样
                batch, weights, tree_idxs = per.memory.sample(BATCH_SIZE)
                
                # 计算TD误差
                loss, td_error = per.update(batch, weights=weights)

                # 更新优先级
                per.memory.update_priorities(tree_idxs, td_error.numpy())

            if(not hit_flag): #没有打中的先把轨迹存上
                trajectory_m[i, :] = (xm_now, ym_now, vm_now, theta_m_now, action_radio_hindsight)
                trajectory_t[i, :] = (xt_now, yt_now, vt_now, theta_t_now)

            hit_finally_flag = hit_flag
            state = state_

        if not hit_finally_flag:
            trajectory_t_all.append(trajectory_t)
            trajectory_m_all.append(trajectory_m)

                        # 和过去同样没打中的目标飞行器轨迹匹配
            print(len(trajectory_m_all))
            print(len(trajectory_t_all))


            trajectory_m_hindsight = trajectory_m_all[-1] # 最新的导弹轨迹，和过去的所有目标轨迹匹配
            for j in range(len(trajectory_t_all)):
                trajectory_t_hindsight = trajectory_t_all[j] # 第j条轨迹
                for k in range(horizon):
                    xm_hindsight = trajectory_m_hindsight[k][0]
                    ym_hindsight = trajectory_m_hindsight[k][1]
                    action_hindsight = trajectory_m_hindsight[k][4]
                    xt_hindsight = trajectory_t_hindsight[k][0]
                    yt_hindsight = trajectory_t_hindsight[k][1]
                    if math.sqrt(math.pow((xt_hindsight - xm_hindsight), 2) + math.pow((yt_hindsight - ym_hindsight), 2)) <= 3.: #匹配成功
                        count += 1
                        env_hindsight = HindsightEnv(trajectory_t_hindsight[:,0], trajectory_t_hindsight[:,1], trajectory_t_hindsight[:,2], trajectory_t_hindsight[:,3], trajectory_m[:,0], trajectory_m[:,1], trajectory_m[:,2], trajectory_m[:,3], Initial_heading_angle_receive, distance_maneuver, Value_direction_maneuver, Value_target_acceleration, horizon)
                        env_hindsight._build_env()
                        state_hindsight = env_hindsight._step(k)
                        state_hindsight_ = env_hindsight._step(k+1)
                        per.memory.add((state_hindsight, action_hindsight, REWARD, state_hindsight_, True))

                        break

    print(count)
    print(per.memory.count)


    # dqn.writer.close()
    torch.save(per.model.state_dict(), '20230603_per' +
               str(main_loop_size) + "_1"+'.pkl')
    print('Over')




