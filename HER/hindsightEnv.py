# 利用 xm_now, ym_now, vm_now, xt_now, yt_now, vt_now 计算当前时刻的状态
# 用于在匹配成功能打中的导弹轨迹和目标飞行物轨迹之后 虚假地计算当前的状态

import numpy as np
import math


class HindsightEnv(object):
    # 输入环境的初始信息
    def __init__(
                self,
                xt_hindsight,  # 匹配成功的目标轨迹
                yt_hindsight,
                vt_hindsight,
                theta_t_hindsight,
                xm_hindsight,  # 匹配成功的导弹轨迹
                ym_hindsight,
                vm_hindsight,
                theta_m_hindsight,
                initial_heading,
                distance_maneuver,
                Value_direction_maneuver,
                Value_target_acceleration,
                horizon
                 ):
        self.xt = [xt_hindsight[0]]
        self.yt = [yt_hindsight[0]]
        self.vt = [vt_hindsight[0]]
        self.theta_t = [theta_t_hindsight[0]]
        self.xm = [xm_hindsight[0]]
        self.ym = [ym_hindsight[0]]
        self.vm = [vm_hindsight[0]]
        self.theta_m = [theta_m_hindsight[0]]
        self.xm_hindsight = xm_hindsight
        self.ym_hindsight = ym_hindsight
        self.vm_hindsight = vm_hindsight
        self.theta_m_hindsight = theta_m_hindsight
        self.xt_hindsight = xt_hindsight
        self.yt_hindsight = yt_hindsight
        self.vt_hindsight = vt_hindsight
        self.theta_t_hindsight = theta_t_hindsight

        self.initial_heading = initial_heading
        self.distance_maneuver = distance_maneuver
        self.Value_direction_maneuver = Value_direction_maneuver
        self.Value_target_acceleration = Value_target_acceleration
        self.horizon = horizon
        self.count_t = 0 # 计数走过了多少个时间步
        # self.goal = np.array([10])
        self.jump_flag = True
        self._build_env()


    # 构建环境最初始的状态
    def _build_env(self):
        self.vtx0 = self.vt[0] * math.cos(0 * math.pi / 180)  # 目标初始速度x轴分量
        self.vty0 = self.vt[0] * math.sin(0 * math.pi / 180)  # 目标初始速度y轴分量
        self.theta_t0 = self.theta_t_hindsight[0]  # 目标初始弹道倾角 弧度
        self.theta_t = [self.theta_t0]
        
        self.vmx0 = self.vm[0] * math.cos(self.initial_heading * math.pi / 180)  # 导弹速度x轴分量
        self.vmy0 = self.vm[0] * math.sin(self.initial_heading * math.pi / 180)  # 导弹速度y轴分量
        self.theta_m0 = self.theta_m_hindsight[0] # 导弹初始弹道倾角 弧度
        self.theta_m = [self.theta_m0]

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
        self.t_inter = 0.01  # 仿真周期10ms
        self.time_t0 = [0.0]

        # 视线转率的二阶动态特性
        self.am0 = self.qdgg0
        self.am_dot0 = 0
        self.am = [self.am0]
        self.am_dot = [self.am_dot0]
        self.t_period_1 = 1 / 8
        self.sigma_1 = 0.5


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

    def _step(self, k):
        # 当前的导弹和目标的位置与速度信息

        for i in range(k):

            xm_now = self.xm[-1]
            ym_now = self.ym[-1]
            vm_now = self.vm[-1]
            xt_now = self.xt[-1]
            yt_now = self.yt[-1]
            vt_now = self.vt[-1]

            self.q.append(math.atan((yt_now - ym_now) / (xt_now - xm_now)))
            self.eng.append(self.theta_m[-1] - self.q[-1])
            self.eng_t.append(self.theta_t[-1] - self.q[-1])
            self.r.append(math.sqrt(math.pow((xt_now - xm_now), 2) + math.pow((yt_now - ym_now), 2)))



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


            self.vm.append(self.vm_hindsight[i+1])
            self.theta_m.append(self.theta_m_hindsight[i+1])
            self.time_t0.append(self.time_t0[-1] + self.t_inter)  # 实时计算时间
            self.xm.append(self.xm_hindsight[i+1])
            self.ym.append(self.ym_hindsight[i+1])
            self.theta_t.append(self.theta_t_hindsight[i+1])
            self.vt.append(self.vt_hindsight[i+1])
            self.xt.append(self.xt_hindsight[i+1])
            self.yt.append(self.yt_hindsight[i+1])

            if self.jump_flag:  # 当第一步时，初始值存在，跳过
                self.jump_flag = False
            else:
                self.v_rela.append(math.sqrt(
                    math.pow(vt_now, 2) + math.pow(vm_now, 2) - 2 * vm_now * vt_now * math.cos(self.theta_t[-2] +
                                                                                            self.theta_m[-2])))
        


        # 返回state
        state = np.array([self.r[-1], self.v_rela[-1], self.q[-1], self.qd_erjie[-1], self.vm[-1], self.theta_m[-1]])
       
        return state