import numpy as np
import math
import matplotlib.pyplot as plt
import time
from testEnv import simulitiveEnv
from hindsightEnv import HindsightEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
# Hyper Parameters
HINDSIGHT_SIZE = 4 # 存储了8个量用于 Hindsight地计算state (xm,ym,vm,theta_m,xt,yt,vt,theta_t)

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.1               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 1000  # 参数更新间隔
MEMORY_CAPACITY = 10000     # 经验池容量

REWARD = 0.1

N_ACTIONS = 25 # action的维度 N1(2,3,4,5,6) \times N2(0,0.5,1,1.5,2)
N_STATES = 6   # state的维度 (r,v_rela,q,q^,v,\theta_m)
LOSS = []



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)     # 输入层将状态嵌入作为输入 并产生128维输出至隐藏层
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(128, N_ACTIONS)    # 输出层接受隐藏层的输出值并针对可能的动作值评估其Q价值 并且选择价值最高的动作作为输出
        self.out.weight.data.normal_(0, 0.1)   # initialization


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value # 返回动作价值 Q


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # 初始化当前网络和目标网络

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2 + 1))     # 初始化经验池  【N_STATES * 2 + 2 + 1】 是为了EBU而打造？？
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # 利用 Adam 进行参数优化
        self.loss_func = nn.MSELoss() # 均方误差 预测值与真实值之差的平方和的平均值

        #parameters for training
        self.batch_num = 0
        self.batch_count = 0
        self.epi_len = 0
        self.epi_state = None       # s
        self.epi_actions = None     # a
        self.epi_rewards = None     # r
        self.epi_state_ = None      # s'

    def choose_action(self, x): # \epsilon-greedy选择动作
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() > EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]   # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action

    def choose_action_greedy(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]   # return the argmax index
        return action

    def store_transition(self, s, a, r, s_, done): # 存储 transition(s,a,r,s')
        transition = np.hstack((s, [a, r], s_, done))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        # 间歇性更新目标网络参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: 
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 随机小批量从经验池中采样
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]) # s
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)) # a
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]) # r
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES-1:-1]) # s'

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        LOSS.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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

    log_dir = 'log\dqn1'
    logger = SummaryWriter(logdir=log_dir)

    np.random.seed(0)
    dqn = DQN()
    # 先初始化环境，初始一个action
    xt0 = 6 * math.pow(10, 3) # 目标初始位置 x
    yt0 = 8 * math.pow(10, 3) # 目标初始位置 y
    vt0 = 300.0 # 目标初始速度
    xm0 = 0.0 # 导弹初始位置 x
    ym0 = 11000.0 # 导弹初始位置 y
    vm0 = 1000.0 # 导弹初始速度
    Initial_heading_angle_receive = -25 # 导弹初始弹道倾角
    distance_maneuver = 2 #这个代表开始运动了2m之后，就开始各个方向机动了。
    Value_direction_maneuver = np.random.randint(1, 4) # 使用Hindsight之后就只能采取随机机动方式
    # Value_direction_maneuver = 2
    Value_target_acceleration = 2 # 目标初始加速度
    arg = (xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver, Value_direction_maneuver,
           Value_target_acceleration)
    main_loop_size = 120
    sample_loop_size = 120
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


    for i_epi in range(sample_loop_size):

        print(i_epi)
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

            action_radio = dqn.choose_action(state)
            action = env_simple.choose_action_RL(action_radio)  # 将动作索引转变为实际的动作

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
                dqn.store_transition(state, action_radio, reward, state_, done)

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

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
                        dqn.store_transition(state_hindsight, action_hindsight, REWARD, state_hindsight_, True)

                        break

    print(count)
    print(dqn.memory_counter)
    

    # dqn.writer.close()
    torch.save(dqn.eval_net.state_dict(), '20230525_hindsight_dqn' +
               str(main_loop_size) +'.pkl')
    print('Over')

