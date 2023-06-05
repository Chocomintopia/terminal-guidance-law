import numpy as np
import pandas as pd
import math
from testEnv import simulitiveEnv
from per_train import Normalizer, PER
import torch
import random

from tensorboardX import SummaryWriter

log_dir1 = 'log\per1_test1'
logger1 = SummaryWriter(logdir=log_dir1)

log_dir2 = 'log\per1_test2'
logger2 = SummaryWriter(logdir=log_dir2)

log_dir3 = 'log\per1_test3'
logger3 = SummaryWriter(logdir=log_dir3)

log_dir4 = 'log\per1_test4'
logger4 = SummaryWriter(logdir=log_dir4)


def restore_parameters(network, net_name):
    """提取网络中的参数"""
    network.load_state_dict(torch.load(net_name))


def test(maneuver_mode, test_num):
    xt0 = 6 * math.pow(10, 3)
    yt0 = 8 * math.pow(10, 3)
    vt0 = 300.0
    xm0 = 0.0
    ym0 = 11000.0
    vm0 = 1000.0
    Initial_heading_angle_receive = -25
    distance_maneuver = 2 #这个代表开始运动了2m之后，就开始各个方向机动了。
    Value_direction_maneuver = 1
    Value_target_acceleration = 2 # 原来是5. 训练时候 目标加速度是2，等会可以更改试试
    arg = (xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver, Value_direction_maneuver,
           Value_target_acceleration)
    xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver, Value_direction_maneuver, Value_target_acceleration = arg
    env = simulitiveEnv(xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver,
                        Value_direction_maneuver, Value_target_acceleration)
    state = env.reset()
    num_inputs = 6
    horizon = 700
    ep_reward = 0
    hit_num3 = 0
    ZEM_Result = []
    aRL = []
    aPN = []
    ZEM_initial_RL = []
    ZEM_initial_PN = []
    theta_m_RL = []
    theta_m_PN = []
    u_OGL_real_RL = []
    u_OGL_real_PN = []

    ZEM_zero_half = []
    ZEM_half_two = []
    ZEM_two_threehalf = []
    ZEM_threehalf_five = []
    ZEM_five_ten = []

    relative_speed_RL = []
    relative_speed_PN = []
    for i in range(test_num):
        xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver, Value_direction_maneuver, Value_target_acceleration = arg
        Initial_heading_angle_receive = -60
        if maneuver_mode == 4:
            Value_direction_maneuver = np.random.randint(1, 4)  # 随机选择三种机动方式
        else:
            Value_direction_maneuver = maneuver_mode
        Value_target_acceleration = 5
        env = simulitiveEnv(xt0, yt0, vt0, xm0, ym0, vm0, Initial_heading_angle_receive, distance_maneuver,
                            Value_direction_maneuver, Value_target_acceleration)
        state = env.reset()
        aRL = []
        done = False
        interrupt_flag = False
        num_plays = 1.
        reward_evaluation = 0
        normalizer = Normalizer(num_inputs)
        while not done and num_plays < horizon and not interrupt_flag:
            normalizer.observe(state)
            state = normalizer.normalize(state)
            action_index = per.choose_action_greedy(state)
            action = env.choose_action_RL(action_index)
            aRL.append(action)
            state, reward, done, interrupt_flag = env._step(action)
            # print(state)

            reward_evaluation += reward
            num_plays += 1
            if num_plays == horizon or done or interrupt_flag:
                # print('=======================================')
                # print('episode:', i)
                # print('Initial_heading_angle:', Initial_heading_angle_receive)
                if done:
                    # print('This episode has hit the target')
                    hit_num3 += 1
                # if interrupt_flag or num_plays == horizon:
                    # print('This episode has not hit the target')
                if i == 0:
                    ZEM_initial_PN = env.ZEM_initial
                    theta_m_PN = env.theta_m
                    u_OGL_real_PN = env.u_OGL_real
                    relative_speed_PN = env.v_relative
                else:
                    ZEM_initial_RL = env.ZEM_initial
                    theta_m_RL = env.theta_m
                    u_OGL_real_RL = env.u_OGL_real
                    relative_speed_RL = env.v_relative
        if env.zem_relative[-1] < 50:
            ZEM_Result.append(env.zem_relative[-1])
        # print('ZEM：', env.zem_relative[-1])
        # print('=======================================')

    ZEM_Result = np.array(ZEM_Result)

    for i in range(len(ZEM_Result)):
        if maneuver_mode == 1:
            logger1.add_scalar('miss distance', ZEM_Result[i], i)
        if maneuver_mode == 2:
            logger2.add_scalar('miss distance', ZEM_Result[i], i)
        if maneuver_mode == 3:
            logger3.add_scalar('miss distance', ZEM_Result[i], i)
        if maneuver_mode == 4:
            logger4.add_scalar('miss distance', ZEM_Result[i], i)

            
    print("个数：", ZEM_Result.size)
    print("脱靶量统计结果：")
    print('Min:', min(ZEM_Result), ',', 'Max:', max(ZEM_Result), ',', 'mean:', ZEM_Result.mean(), 'std:', ZEM_Result.std())
    result1 = [maneuver_mode, min(ZEM_Result), max(ZEM_Result), ZEM_Result.mean(),ZEM_Result.std()]

    ZEM_zero_half = 0
    ZEM_half_two = 0
    ZEM_two_threehalf = 0
    ZEM_threehalf_five = 0
    ZEM_five_ten = 0

    for zem in ZEM_Result:
        if zem >= 0 and zem <= 0.5:
            ZEM_zero_half += 1
        elif zem > 0.5 and zem <= 3.0:
            ZEM_half_two += 1
        elif zem > 3.0 and zem <= 5.0:
            ZEM_two_threehalf += 1
        elif zem > 5.0 and zem <= 20.0:
            ZEM_threehalf_five += 1
        elif zem > 20.0:
            ZEM_five_ten += 1

    print('0-0.5:', ZEM_zero_half, ',', '0.5-3:', ZEM_half_two, ',', '3-5:', ZEM_two_threehalf, ',', '5-20:',
          ZEM_threehalf_five, ',', '20+:', ZEM_five_ten)
    result2 = [maneuver_mode,
               '{:.2%}'.format(ZEM_zero_half/len(ZEM_Result)),
               '{:.2%}'.format(ZEM_half_two/len(ZEM_Result)),
               '{:.2%}'.format(ZEM_two_threehalf/len(ZEM_Result)),
               '{:.2%}'.format(ZEM_threehalf_five/len(ZEM_Result)),
               '{:.2%}'.format(ZEM_five_ten/len(ZEM_Result)),
               ]
    return result1, result2


if __name__ == '__main__':
    per = PER()
    # 提取网络中参数

    restore_parameters(per.model, '4final_20230528_per120_1.pkl')


    maneuver_list = [1, 2, 3, 4]
    header1 = ['type', 'Min', 'Max', 'Mean', 'Stdev']
    header2 = ['type', '[0, 0.5]', '[0.5, 3]', '[3, 5]', '[5, 20]', '[20,  ]']
    df1 = pd.DataFrame(columns=header1)
    df1.to_csv('result/per_ZEM.csv', index=False, header=header1)
    df1 = pd.DataFrame(columns=header2)
    df1.to_csv('result/per_percentage.csv', index=False, header=header2)
    for maneuver_mode in maneuver_list:
        data1, data2 = test(maneuver_mode, 100) # 测试100次
        df1 = pd.DataFrame(data=np.around([data1], 2), columns=header1)
        df1.to_csv('result/per_ZEM.csv', mode='a', index=False, header=False)
        df2 = pd.DataFrame(data=[data2], columns=header2)
        df2.to_csv('result/per_percentage.csv', mode='a', index=False, header=False)
    print('finished!')
