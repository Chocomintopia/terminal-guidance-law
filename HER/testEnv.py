"""
一个环境
"""

# 需要改，轨迹还是要恢复原来的done，也就是会提前结束，只是增加了hit_flag

import numpy as np
import math
import airdensity
import table
import matplotlib.pyplot as plt

class simulitiveEnv(object):
    """
    仿真环境
    """
    # 输入环境的初始信息
    def __init__(
                self,
                xt0,  # 目标初始信息
                yt0,
                vt0,
                xm0,  # 导弹初始信息
                ym0,
                vm0,
                initial_heading,
                distance_maneuver,
                Value_direction_maneuver,
                Value_target_acceleration,
                horizon
                 ):
        self.xt = [xt0]
        self.yt = [yt0]
        self.vt = [vt0]
        self.xm = [xm0]
        self.ym = [ym0]
        self.vm = [vm0]
        self.initial_heading = initial_heading
        self.distance_maneuver = distance_maneuver
        self.Value_direction_maneuver = [Value_direction_maneuver]
        self.Value_target_acceleration = Value_target_acceleration
        self.horizon = horizon
        self.count = 0
        # self.goal = np.array([10])
        self.hit_flag = False
        self.jump_flag = True
        self._build_env()


    # 构建环境最初始的状态
    def _build_env(self):
        self.hit_flag = False
        self.count = 0

        self.vtx0 = self.vt[0] * math.cos(0 * math.pi / 180)  # 目标初始速度x轴分量
        self.vty0 = self.vt[0] * math.sin(0 * math.pi / 180)  # 目标初始速度y轴分量
        self.theta_t0 = math.atan(self.vty0 / self.vtx0) - math.pi  # 目标初始弹道倾角 弧度
        self.theta_t = [self.theta_t0]
        self.initial_angle_ideal = -(math.asin(self.vt[0] * math.sin(math.atan((self.ym[0] - self.yt[0]) /
                (self.xt[0] - self.xm[0]))) / self.vm[0]) + math.atan((self.ym[0] - self.yt[0]) / (self.xt[0] - self.xm[0]))) \
                              * 180 / math.pi  # 初始零控脱靶量
        self.vmx0 = self.vm[0] * math.cos(self.initial_heading * math.pi / 180)  # 导弹速度x轴分量
        self.vmy0 = self.vm[0] * math.sin(self.initial_heading * math.pi / 180)  # 导弹速度y轴分量
        self.theta_m0 = math.atan(self.vmy0 / self.vmx0)  # 导弹初始弹道倾角 弧度
        self.theta_m = [self.theta_m0]

        # 初始气动参数 推力 质量信息
        self.rou0, self.v_sound0 = airdensity.airdensityFun(self.ym[0])  # 初始空气密度和声速
        self.rou = []
        self.Mach0 = self.vm[0] / self.v_sound0  # 初始马赫数
        self.S_xsf = 0.2  # 导弹的特征面积
        self.mf = 400  # 导弹的质量
        self.vm_dot = [-25]  # 导弹速度的初始变化率
        self.q0 = math.atan((self.yt[0] - self.ym[0]) / (self.xt[0] - self.xm[0]))  # 初始弹目视线角
        self.q = []  # 初始弹目视线角数组
        self.v_rela0 = math.sqrt(
            math.pow(self.vt[0], 2) + math.pow(self.vm[0], 2) - 2 * self.vm[0] * self.vt[0] *
            math.cos(self.theta_t0 + self.theta_m0))  # 初始相对速度
        self.v_rela = [self.v_rela0]  # 相对速度数组
        self.r0 = math.sqrt(math.pow((self.xt[0] - self.xm[0]), 2) + math.pow((self.yt[0] - self.ym[0]), 2))  # 初始弹目相对距离
        self.r = []
        self.eng0 = self.theta_m0 - self.q0  # 导弹初始弹道倾角与弹目视线角的夹角 弧度
        self.eng_t0 = self.theta_t0 - self.q0  # 目标初始弹道倾角与弹目视线角的夹角 弧度
        self.eng = []
        self.eng_t = []
        # 初始视线转率
        self.qdgg0 = (self.vt[0] * math.sin(self.eng_t0) - self.vm[0] * math.sin(self.eng0)) / self.r0
        self.qdgg = [self.qdgg0]  # 视线转率数组,第一项为初始视线转率
        self.ZEM_initial0 = math.pow(self.r0, 2) * self.qdgg0 / self.v_rela0  # 初始脱靶量
        self.ZEM_initial = [self.ZEM_initial0]  # 脱靶量数组
        self.time = 150  # 150s
        self.t_inter = 0.01  # 仿真周期10ms
        self.control_sum = 0.0
        self.time_t0 = [0.0]
        self.alpha_max = 35 * math.pi / 180  # 允许的最大攻角
        self.var_Navigation_1 = 4  # 导航比N1
        self.var_Navigation_2 = 1  # 导航比N2

        # 视线转率的二阶动态特性
        self.am0 = self.qdgg0
        self.am_dot0 = 0
        self.am = [self.am0]
        self.am_dot = [self.am_dot0]
        self.t_period_1 = 1 / 8
        self.sigma_1 = 0.5

        # 导弹的二阶动态特性
        self.amu = [0.0]
        self.amu_dot = [0.0]
        self.t_period_2 = 1 / 15
        self.sigma_2 = 0.65

        # 目标的一阶动态特性
        self.var_tao2 = 0.5
        self.at = [0]
        self.at_dot = []

        # 一阶低通滤波器对视线转率进行滤波估计
        self.var_tao3 = 0.4
        self.qd_erjie = [self.qdgg0]
        self.qd_erjie_dot = []

        # 高斯白噪声的一阶滤波
        self.var_k = 1
        self.var_eng = 0.02
        self.var_tao1 = 1 / 16
        self.RWzj = [0.0]
        self.RWzj_dot = []

        # 参数数组定义
        self.t_flag = int(self.time / self.t_inter * 0 + 100000)
        self.C_Lafa = []  # 气动参数
        self.C_K = []  # 气动参数
        self.C_D = []  # 气动参数
        self.m = []
        self.vx_relative = []  # x轴方向弹目相对速度
        self.vy_relative = []  # y轴方向弹目相对速度
        self.v_relative = []  # 弹目相对速度
        self.rx_relative = []  # x轴方向弹目相对距离
        self.ry_relative = []  # y 轴方向弹目相对距离
        self.cos_relative = []
        self.zem_relative = []  # 脱靶量
        self.rd = []  # 弹目距离变化率
        self.qdc = []  # 理想视线转率
        self.time_tf = []  # 计算终端时间
        self.tgo = []  # 剩余飞行时间
        # 角噪声和目标闪烁数据
        self.RWzj_c = []
        self.RWz = []
        self.qd_noise = []
        # 视线转率的二阶动态特性数据
        self.am_c = []
        self.am_dot2 = []
        self.qd_erjie_c = []
        self.qd = []
        self.at_axis = []
        self.atc_normal = []
        # 目标的一阶动态特性
        self.at_c = []
        self.at_normal = []
        # 增广比例制导律，对目标的机动和重力作用进行补偿
        self.var_uc = []
        # 导弹的二阶动态特性
        self.amu_c = []
        self.amu_dot2 = []
        self.var_u = []
        self.u_OGL = []
        self.alpha = []
        self.u_OGL_real = []

    def _step(self, action):
        # 当前的导弹和目标的位置与速度信息
        xm_now = self.xm[-1]
        ym_now = self.ym[-1]
        vm_now = self.vm[-1]
        xt_now = self.xt[-1]
        yt_now = self.yt[-1]
        vt_now = self.vt[-1]

        rou_now, v_sound_now = airdensity.airdensityFun(ym_now)  # 实时计算空气密度和声速
        self.rou.append(rou_now)
        self.C_Lafa.append(table.table_aerocoeff_Ma_lookup(vm_now / v_sound_now, 4))  # 气动参数
        self.C_K.append(table.table_aerocoeff_Ma_lookup(vm_now / v_sound_now, 3))  # 气动参数
        self.C_D.append((table.table_aerocoeff_Ma_lookup(vm_now / v_sound_now, 2)))  # 气动参数
        self.m.append(self.mf)
        self.q.append(math.atan((yt_now - ym_now) / (xt_now - xm_now)))
        self.eng.append(self.theta_m[-1] - self.q[-1])
        self.eng_t.append(self.theta_t[-1] - self.q[-1])
        self.r.append(math.sqrt(math.pow((xt_now - xm_now), 2) + math.pow((yt_now - ym_now), 2)))

        self.vx_relative.append(vt_now * math.cos(self.theta_t[-1]) - vm_now * math.cos(self.theta_m[-1]))  # 弹目x轴相对速度
        self.vy_relative.append(vt_now * math.sin(self.theta_t[-1]) - vm_now * math.sin(self.theta_m[-1]))  # 弹目y轴相对速度
        self.v_relative.append(math.sqrt(math.pow(self.vx_relative[-1], 2) + math.pow(self.vy_relative[-1], 2)))  # 弹目相对速度
        self.rx_relative.append(xt_now - xm_now)  # x轴相对距离
        self.ry_relative.append(yt_now - ym_now)  # y轴相对距离
        self.cos_relative.append(
            (self.vx_relative[-1] * self.rx_relative[-1] +
             self.vy_relative[-1] * self.ry_relative[-1]) / (self.r[-1] * self.v_relative[-1]+1))
        self.zem_relative.append(self.r[-1] * math.sqrt(1 - math.pow(self.cos_relative[-1], 2)))  # 脱靶量

        if not self.hit_flag: # 没有打中过
            self.hit_flag = self._is_hit_state()  # 判断是否打中 获得一个bool值,如果导弹与目标距离<5m，代表打中，done是true
        
        if self.hit_flag: # 打中了
            self.count += 1
        
        done = self._is_end_state() or self._is_hit_state() # 判断是否是中止状态

        self.rd.append(vt_now * math.cos(self.eng_t[-1]) - vm_now * math.cos(self.eng[-1]))  # 弹目距离变化率
        self.qdc.append((vt_now * math.sin(self.eng_t[-1]) - vm_now * math.sin(self.eng[-1])) / self.r[-1])  # 理想视线转率
        self.time_tf.append(self.time_t0[-1] + abs(self.r[-1] / self.rd[-1]))  # 实时计算终端时间
        self.tgo.append(abs(self.r[-1] / self.rd[-1]))  # 实时计算剩余飞行时间

        # 角噪声和目标闪烁
        self.RWzj_c.append(np.random.randn())
        self.RWzj_dot.append((self.RWzj_c[-1] - self.RWzj[-1]) / self.var_tao1)
        """
        注意此处的Rwzj数组在迭代时使用了自己的值，所以在接下来使用这个值的时候，下标应为-2，既数组
        的倒数第二个值，因为倒数第一个值是新添加进去的，出现这种情况的原因是Rwzj在一开始的时候
        是存在初始值的，在接下来也要注意这种情况
        """
        self.RWzj.append(self.RWzj[-1] + self.RWzj_dot[-1] * self.t_inter)
        self.RWz.append(np.random.randn())
        self.qd_noise.append(self.var_k * (self.RWz[-1] * self.var_eng * math.pow(10, -7) * math.pow(self.r[-1], 2) +
                                           self.RWzj[-2] / self.r[-1]))

        # 视线转率的二阶动态特性
        self.am_c.append(self.qdc[-1] + self.qd_noise[-1])
        self.am_dot2.append((self.am_c[-1] - 2 * self.t_period_1 * self.sigma_1 * self.am_dot[-1] - self.am[-1]) /
                            math.pow(self.t_period_1, 2))
        # am_dot有初始值，在迭代中使用了自己，情况同上
        self.am_dot.append(self.am_dot[-1] + self.am_dot2[-1] * self.t_inter)
        self.am.append(self.am[-1] + self.am_dot[-2] * self.t_inter)
        self.qd_erjie_c.append(self.am[-1])

        # 采用一阶低通滤波器对视线转率进行滤波
        self.qd_erjie_dot.append((self.qd_erjie_c[-1] - self.qd_erjie[-1]) /
                                 (self.var_tao3 * self.tgo[-1] / self.time_tf[-1]))
        # qd_erjie有初始值，在迭代中使用了自己，情况同上
        self.qd_erjie.append(self.qd_erjie[-1] + self.qd_erjie_dot[-1] * self.t_inter)
        self.qd.append(self.qd_erjie[-2])  # 经过低通滤波器后的到的视线转率
        self.at_axis.append(
            -0.01 * math.pow(self.time_t0[-1], 3) + 0.22 * math.pow(self.time_t0[-1], 2) -
            1.2 * self.time_t0[-1] + 9.4)  # 目标轴向加速度
        if self.r[-1] >= self.distance_maneuver: # 超过distance_maneuver这个距离，开始机动。
            if self.Value_direction_maneuver[-1] == 1: # 机动方式
                self.atc_normal.append(-9.8 * (self.Value_target_acceleration - 1))  # 向上机动
            elif self.Value_direction_maneuver[-1] == 2:
                self.atc_normal.append(9.8 * (self.Value_target_acceleration - 1))  # 向下机动
            elif self.Value_direction_maneuver[-1] == 3:
                self.atc_normal.append(9.8 * (self.Value_target_acceleration - 1) *
                                        np.sign(math.sin(1 * self.time_t0[-1])))  # 上下机动
        else:
            self.atc_normal.append(0)

        # 目标的一阶动态特性
        self.at_c.append(self.atc_normal[-1])
        self.at_dot.append((self.at_c[-1] - self.at[-1]) / self.var_tao2)
        # qd_erjie有初始值，在迭代中使用了自己，情况同上
        self.at.append(self.at[-1] + self.at_dot[-1] * self.t_inter)
        self.at_normal.append(self.at[-2])  # 这几个值一直是0

        # 制导律 此时的制导律来源于给出的action
        self.var_uc.append(action)

        # 导弹的二阶动态特性
        self.amu_c.append(self.var_uc[-1])
        self.amu_dot2.append(1 / self.t_period_2 * 1 / self.t_period_2 * self.amu_c[-1] - 2 * self.sigma_2 * 1 /
                             self.t_period_2 * self.amu_dot[-1] - 1 / self.t_period_2 * 1 / self.t_period_2 *
                             self.amu[-1])
        # amu_dot有初始值，在迭代中使用了自己，情况同上
        self.amu_dot.append(self.amu_dot[-1] + self.amu_dot2[-1] * self.t_inter)
        # amu有初始值，在迭代中使用了自己，情况同上
        self.amu.append(self.amu[-1] + self.amu_dot[-2] * self.t_inter)
        self.var_u.append(self.amu[-2])
        self.u_OGL.append(self.var_u[-1])
        if abs(self.u_OGL[-1]) > 25 * 9.8:
            self.u_OGL[-1] = 25 * 9.8 * np.sign(self.u_OGL[-1])
        self.alpha.append(self.u_OGL[-1] * self.m[-1] / (self.C_Lafa[-1] * 0.5 * self.rou[-1] *
                                                         math.pow(vm_now, 2) * self.S_xsf))  # 将过载指令转换为攻角指令
        if abs(self.alpha[-1]) > self.alpha_max:
            self.alpha[-1] = self.alpha_max * np.sign(self.alpha[-1])

        # 参数迭代
        # vm_dot有初始值，在添加了最新一项后使用了倒数第二项，情况同上
        self.vm_dot.append(
            (-(self.C_D[-1] + self.C_K[-1] * math.pow(self.C_Lafa[-1], 2) * math.pow(self.alpha[-1], 2)) *
                0.5 * self.rou[-1] * math.pow(vm_now, 2) * self.S_xsf)
            / self.m[-1] - 9.8 * math.sin(self.theta_m[-1]))
        self.vm.append(vm_now + self.vm_dot[-2] * self.t_inter)
        self.u_OGL_real.append(
            (self.C_Lafa[-1] * 0.5 * self.rou[-1] * math.pow(vm_now, 2) * self.S_xsf) * self.alpha[-1] /
            self.m[-1])  # 对攻角进行限幅后得到过载指令
        self.theta_m.append(self.theta_m[-1] + (self.u_OGL_real[-1] -
                                                9.8 * math.cos(self.theta_m[-1])) / vm_now * self.t_inter)
        self.control_sum = self.control_sum + abs(self.u_OGL[-1]) * self.t_inter
        self.time_t0.append(self.time_t0[-1] + self.t_inter)  # 实时计算时间
        # theta_m有初始值，在添加了最新一项后使用了倒数第二项，情况同上
        self.xm.append(xm_now + vm_now * math.cos(self.theta_m[-2]) * self.t_inter)
        self.ym.append(ym_now + vm_now * math.sin(self.theta_m[-2]) * self.t_inter)
        self.theta_t.append(self.theta_t[-1] + (self.at_normal[-1] / vt_now) * self.t_inter)
        self.vt.append(vt_now + self.at_axis[-1] * self.t_inter)
        # theta_t有初始值，在添加了最新一项后使用了倒数第二项，情况同上
        self.xt.append(xt_now + vt_now * math.cos(self.theta_t[-2]) * self.t_inter)
        self.yt.append(yt_now + vt_now * math.sin(self.theta_t[-2]) * self.t_inter)
        self.Value_direction_maneuver.append(np.random.randint(1, 4))
        # 瞬时零控脱靶量
        if self.jump_flag:  # 当第一步时，初始值存在，跳过
            self.jump_flag = False
        else:
            self.v_rela.append(math.sqrt(
                math.pow(vt_now, 2) + math.pow(vm_now, 2) - 2 * vm_now * vt_now * math.cos(self.theta_t[-2] +
                                                                                           self.theta_m[-2])))
            self.ZEM_initial.append(math.pow(self.r[-1], 2) * abs(self.qdc[-1]) / self.v_rela[-1])

        # 返回state
        # state = np.array([self.r[-1], self.v_rela[-1], self.q[-1], self.qd_erjie[-1], self.vm[-1], self.theta_m[-1], self.zem_relative[-1]])
        state = np.array([self.r[-1], self.v_rela[-1], self.q[-1], self.qd_erjie[-1], self.vm[-1], self.theta_m[-1]])
        # state = np.array([self.r[-1], self.v_rela[-1], self.q[-1], self.qd_erjie[-1], self.theta_m[-1], self.zem_relative[-1]])
        # state = np.array([self.r[-1], self.v_rela[-1], self.q[-1], self.qd_erjie[-1], self.vm[-1], self.theta_m[-1], self.xm[-1], self.ym[-1], self.xt[-1], self.yt[-1]])
        # state = np.array([self.r[-1], self.v_rela[-1]])
        # state = np.array([self.r[-1]])



        # 二值稀疏奖励
        if self.count == 1: # 第一次计算出打中的时候给一个奖励 以后再算出来就不给了
            reward = 1
        else:
            reward = 0


        # #这个是一些标志位的判断
        # if not done:
        #     self.interrupt_flag = False
        # else:
        #     self.done_flag = 0




        # # 当相对距离开始连续变大时 则没打中 停止仿真
        # if len(self.r) > 3:
        #     if self.r[-1] > self.r[-2]:
        #         self.done_flag += 1
        # if not done and self.done_flag >= 2:    #interrupt_flag代表着这次打偏了，没打中，也会结束
        #     self.interrupt_flag = True
        #     self.done_flag = 0
       
        return state, reward, done, self.hit_flag

        
    def render(self):
        plt.plot(self.xm, self.ym, 'r', self.xt, self.yt, 'g')
        plt.show()
        
    # # 判断当前状态是否是episode中止状态
    # def _is_end_state(self):
    #     if len(self.r) > 1:  # 当r数组中的数据两项以上时可以进行判断
    #         # if self.r[-1] > self.r[-2]:   # 此条件是之前终止循环的条件
    #         # if self.r[-1] <= 10:   # 此条件是之前终止循环的条件
    #         if self.r[-1] <= 3:  # 此条件是之前终止循环的条件
    #             """
    #             i = i - 1
    #             print('zem_relative=' + str(zem_relative[i]))  # 输出脱靶量
    #             print(str(i))
    #             break
    #             """
    #             return True
    #         else:
    #             return False
    #     else:  # 此时r数组中的数据还不足两项
    #         return False

    def _is_hit_state(self): # 判断打没打中
        if len(self.r) > 1:  # 当r数组中的数据两项以上时可以进行判断
            if self.r[-1] <= 3:  # 此条件是之前终止循环的条件
                return True
            else:
                return False
        else:  # 此时r数组中的数据还不足两项
            return False
    
    def _is_end_state(self): # 失败的轨迹 horizon 全部坚持打完才结束
        if len(self.r) <= self.horizon:
            return False
        else:
            return True

    def _reset(self):
        # self._build_env()
        action = 0
        state,_,_,_ = self._step(action)
        # state = np.array([self.r0, self.v_rela0])
        return state
    def choose_action(self,i):
        self.var_Navigation_1 = i
        action = (self.var_Navigation_1 * abs(self.rd[-1]) * self.qd[-1] + self.var_Navigation_2 * self.at_normal[-1] * math.cos(self.eng_t[-1])) / math.cos(self.theta_m[-1] - self.q[-1]) + 9.8 * math.cos(self.theta_m[-1])
        return action
    def choose_action_origin(self,i):
        action = i * abs(self.rd[-1]) * self.qd[-1]
        return action
    def choose_action_RL(self,action_index):
        indices_matrix = np.arange(0,25).reshape(5,5)
        index_to_navigation1, index_to_navigation2 = 0, 0
        for i in range(5):
            for j in range(5):
                if indices_matrix[i][j] == action_index:
                    index_to_navigation1, index_to_navigation2 = i, j
        action = ((index_to_navigation1 + 2) * abs(self.rd[-1]) * self.qd[-1] +( index_to_navigation2 / 2) * self.at_normal[
            -1] * math.cos(self.eng_t[-1])) / math.cos(self.theta_m[-1] - self.q[-1]) + 9.8 * math.cos(self.theta_m[-1])
        return action
    def reset(self): #wty
        action = 0
        self._build_env()
        # state = np.array([self.r0, self.v_rela0, self.q0, self.qdgg0, self.vm[0], self.theta_m[0], self.xm[0], self.ym[0], self.xt[0], self.yt[0]])
        # state = np.array([self.r0, self.v_rela0])
        # state = np.array([self.r0])
        # state = np.array([self.r0, self.v_rela0, self.q0, self.qdgg0, self.vm[0], self.theta_m[0]])
        state,_,_,_ = self._step(action) #wty
        return state
    
    # def reward_func(self, state, goal):
    #     if state[0] > goal[0]:
    #         done = False
    #         reward = -1
    #     else:
    #         done = True
    #         reward = 0
    #     return done, reward

#xt0 = 8 * math.pow(10, 3)
#yt0 = 4 * math.pow(10, 3)
#vt0 = 300.0
#xm0 = 0.0
#ym0 = 11000.0
#vm0 = 1400.0
#Initial_heading_angle_receive = -25
#distance_maneuver = 1
#Value_direction_maneuver = 3
#Value_target_acceleration = 2
#env1=simulitiveEnv(xt0,yt0,vt0,xm0,ym0,vm0,Initial_heading_angle_receive,distance_maneuver,Value_direction_maneuver,Value_target_acceleration)
#states = []
#for i in range(1000):
#    if i !=0:
#        action = env1.choose_action()
#    else:
#        action = 1
#    state = env1._step(action)
#    states.append(state)
#    print(state[0][0])
#    if state[-1] == True:
#        print('Done!%d' %i)

#        break
#plt.figure()
#plt.plot(env1.xm, env1.ym)
#plt.xlabel('X轴(m)')
#plt.xlabel('Y轴(m)')
#plt.plot(env1.xt, env1.yt, 'r')
#plt.show()
#print(len(states[54]))














