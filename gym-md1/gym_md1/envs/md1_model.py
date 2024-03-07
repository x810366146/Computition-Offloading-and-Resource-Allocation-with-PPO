import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np


class MD1ModelEnv(gym.Env):
    def __init__(self, render_mode=None, num_queue=4):
        # 定义队列的数量
        self.num_queue = num_queue
        self.time_slot = 600  # 时隙600s
        self.lambda_low = 15  # λ最小15
        self.lambda_high = 25  # λ最大25
        self.Fc_low = 10 * 1024  # 边缘服务器计算能力最小 10T FLOPS
        self.Fc_high = 15 * 1024  # 边缘服务器计算能力最大 20T FLOPS
        self.Fe = 2 * 1024  # 本地用户计算能力为 3T FLOPS
        self.R_rl_low = 5  # 无线带宽最小 10Mbps
        self.R_rl_high = 10  # 无线带宽最大 20Mbps
        self.energy_budget = 10  # 能量预算
        self.energy_cons = 1.5 * 1e-5  # 能量消耗
        self.lyapunovQ = 0
        self.min_partition = 0.
        self.max_partition = 1.0

        self.task_info = [[] for _ in range(self.num_queue)]
        self.task_info[0] = [0.1, 0.4, 3]  # AlexNet 最大容忍时间0.1s 数据量4Mb 所需计算能力 3G FLOPS
        self.task_info[1] = [0.2, 0.15, 30]  # VGG19
        self.task_info[2] = [0.2, 1, 80]  # ResNet101
        self.task_info[3] = [0.15, 0.5, 10]  # YOLOv2

        self.average_delay = 0.
        self.average_energy_consmption = 0.
        self.average_count = 0.


        self.state = None

        # 定义状态空间
        ob_high = np.array(
            [
                self.lambda_high,
                self.lambda_high,
                self.lambda_high,
                self.lambda_high,
                self.R_rl_high,
                self.Fc_high,
                np.finfo(np.float32).max,
            ]
        )
        ob_low = np.array(
            [
                self.lambda_low,
                self.lambda_low,
                self.lambda_low,
                self.lambda_low,
                self.R_rl_low,
                self.Fc_low,
                0
            ]
        )
        self.observation_space = spaces.Box(
            low=ob_low, high=ob_high, dtype=np.float32
        )

        # 定义动作空间
        self.action_space = spaces.Box(
            low=self.min_partition, high=self.max_partition, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        ob_high = np.array(
            [
                self.lambda_high,
                self.lambda_high,
                self.lambda_high,
                self.lambda_high,
                self.R_rl_high,
                self.Fc_high,
                0
            ]
        )
        ob_low = np.array(
            [
                self.lambda_low,
                self.lambda_low,
                self.lambda_low,
                self.lambda_low,
                self.R_rl_low,
                self.Fc_low,
                0
            ]
        )
        self.state = self.np_random.uniform(low=ob_low, high=ob_high, size=(7,))
        self.state[0:self.num_queue] = np.round(self.state[0:self.num_queue])
        self.average_count = 0.
        self.average_delay = 0.
        self.average_energy_consmption = 0.
        self.average_count = 0.
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        lambda_t = [self.state[i] for i in range(self.num_queue)]
        R_wireless = self.state[self.num_queue]
        Fc = self.state[self.num_queue + 1]
        lyapunovQ = self.state[self.num_queue + 2]

        # print("-----------step--------------------")
        # print(f'stete: {self.state}')

        # print(f'action = {action}')

        x_t = action
        fc = self.resource_allocation(x_t)
        E_t = self.compute_energy_consumption(x_t)
        D_t = self.compute_total_time(self.compute_delay(x_t, fc))
        self.compute_average(D_t, E_t)

        V = 100
        reward = -(lyapunovQ * (E_t - self.energy_budget) + V * D_t)
        lyapunovQ = max(lyapunovQ - self.energy_budget + E_t, 0)

        ob_high = np.array(
            [
                self.lambda_high,
                self.lambda_high,
                self.lambda_high,
                self.lambda_high,
                self.R_rl_high,
                self.Fc_high,
                0
            ]
        )
        ob_low = np.array(
            [
                self.lambda_low,
                self.lambda_low,
                self.lambda_low,
                self.lambda_low,
                self.R_rl_low,
                self.Fc_low,
                0
            ]
        )
        self.state = self.np_random.uniform(low=ob_low, high=ob_high, size=(7,))
        self.state[0:self.num_queue] = np.round(self.state[0:self.num_queue])
        self.state[self.num_queue + 2] = lyapunovQ

        return np.array(self.state, dtype=np.float32), reward, False, False, {"average_delay" : self.average_delay, "average_energy_cons" : self.average_energy_consmption}

    def resource_allocation(self, x_t):
        lambda_t = [self.state[k] for k in range(self.num_queue)]
        R_wireless = self.state[self.num_queue]
        Fc = self.state[self.num_queue + 1]

        comp_resource = [x_t[k] * self.task_info[k][2] for k in range(self.num_queue)]  # *每个*任务在边缘侧所需要的计算资源
        total = 0  # 边缘侧一共需要提供的计算资源
        for k in range(self.num_queue):
            total += lambda_t[k] * comp_resource[k]
        if total != 0:
            fc = [lambda_t[k] * comp_resource[k] * Fc / total for k in range(self.num_queue)]  # 按比例分配计算资源
            D_t_star = self.compute_delay(x_t, fc)
            while True:
                # 拷贝必须是拷贝
                D_t = D_t_star[:]
                fc, D_t_star = self.sub_resource_allocation(x_t, fc, D_t_star)
                if self.compute_total_time(D_t) - self.compute_total_time(D_t_star) <= 1e-3:
                    break
            return fc
        else:
            return [0 for k in range(self.num_queue)]
            ## ababcbacadefegdehijhklij
    def sub_resource_allocation(self, x_t, fc, D_t_star):
        lambda_t = [self.state[i] for i in range(self.num_queue)]
        delta_D_minus = [0 for k in range(self.num_queue)]
        delta_D_plus = [0 for k in range(self.num_queue)]

        for k in range(self.num_queue):
            fc_minus = fc[:]
            fc_plus = fc[:]
            fc_minus[k] -= 100
            fc_plus[k] += 100
            D_k_minus = 0
            D_k_plus = 0

            if fc_minus[k] > lambda_t[k] * x_t[k] * self.task_info[k][2]:
                D_k_minus = self.compute_delay_single(x_t, fc_minus, k)
                if D_k_minus > self.task_info[k][0]:
                    D_k_minus = D_t_star[k]
            else:
                D_k_minus = D_t_star[k]
            D_k_plus = self.compute_delay_single(x_t, fc_plus, k)
            delta_D_minus[k] = D_k_minus - D_t_star[k]
            delta_D_plus[k] = D_t_star[k] - D_k_plus

        k_minus, k_plus = 0, 0
        for k in range(self.num_queue):
            if delta_D_minus[k_minus] > delta_D_minus[k]:
                k_minus = k
            if delta_D_plus[k_plus] < delta_D_plus[k]:
                k_plus = k

        if delta_D_plus[k_plus] > delta_D_minus[k_minus]:
            fc[k_plus] += 100
            fc[k_minus] -= 100
            D_t_star = self.compute_delay(x_t, fc)

        return fc, D_t_star

    def compute_delay(self, x_t, fc):
        lambda_t = [self.state[i] for i in range(self.num_queue)]
        R_wireless = self.state[self.num_queue]
        D_t = [0 for i in range(self.num_queue)]  # 每个队列的延迟时间

        for i in range(self.num_queue):
            D_user = 0  # 本地执行时间
            if x_t[i] != 1:  # 1代表全部卸载到边缘服务器
                u_user = self.Fe / (self.task_info[i][2] * (1 - x_t[i]))
                D_user = 1 / u_user + lambda_t[i] / (2 * u_user ** 2 * (1 - lambda_t[i] / u_user))

            D_trans = self.task_info[i][1] * x_t[i] / R_wireless  # 数据传输时间

            D_edge = 0  # 在边缘服务器的执行时间
            if 0 < x_t[i] < 1:
                u_edge = fc[i] / (self.task_info[i][2] * x_t[i])
                u_user = self.Fe / (self.task_info[i][2] * (1 - x_t[i]))
                if u_edge >= u_user:
                    D_edge = 1 / u_edge
                else:
                    D_edge = 1 / u_edge + lambda_t[i] / (2 * u_edge ** 2 * (1 - lambda_t[i] / u_edge))
            elif x_t[i] == 0:
                D_edge = 0
            else:  # x_t[i] == 1
                u_edge = fc[i] / (self.task_info[i][2] * x_t[i])
                D_edge = 1 / u_edge + lambda_t[i] / (2 * u_edge ** 2 * (1 - lambda_t[i] / u_edge))

            D_t[i] = D_user + D_trans + D_edge

        return D_t

    def compute_delay_single(self, x_t, fc, k):
        lambda_t_k = self.state[k]
        R_wireless = self.state[self.num_queue]
        D_t_k = 0

        D_user = 0  # 本地执行时间
        if x_t[k] != 1:  # 1代表全部卸载到边缘服务器
            u_user = self.Fe / (self.task_info[k][2] * (1 - x_t[k]))
            D_user = 1 / u_user + lambda_t_k / (2 * u_user ** 2 * (1 - lambda_t_k / u_user))

        D_trans = self.task_info[k][1] * x_t[k] / R_wireless  # 数据传输时间

        D_edge = 0  # 在边缘服务器的执行时间
        if 0 < x_t[k] < 1:
            u_edge = fc[k] / (self.task_info[k][2] * x_t[k])
            u_user = self.Fe / (self.task_info[k][2] * (1 - x_t[k]))
            if u_edge >= u_user:
                D_edge = 1 / u_edge
            else:
                D_edge = 1 / u_edge + lambda_t_k / (2 * u_edge ** 2 * (1 - lambda_t_k / u_edge))
        elif x_t[k] == 0:
            D_edge = 0
        else:  # x_t[i] == 1
            u_edge = fc[k] / (self.task_info[k][2] * x_t[k])
            D_edge = 1 / u_edge + lambda_t_k / (2 * u_edge ** 2 * (1 - lambda_t_k / u_edge))

        D_t_k = D_user + D_trans + D_edge
        return D_t_k

    def compute_energy_consumption(self, x_t):
        E_t = [0 for k in range(self.num_queue)]
        for k in range(self.num_queue):
            if x_t[k] == 0:
                E_t[k] = 0
            else:
                E_t[k] = self.time_slot * self.energy_cons * self.state[k] * self.task_info[k][2] * x_t[k]

        total = 0
        for k in range(self.num_queue):
            total += E_t[k]
        return total

    def compute_total_time(self, D_t):
        total = 0
        for k in range(self.num_queue):
            total += D_t[k]
        return total

    def compute_average(self, D_t, E_t):
        self.average_count += 1.0
        self.average_delay = ((self.average_count - 1) / self.average_count) * self.average_delay +  D_t / self.average_count
        self.average_energy_consmption = ((self.average_count - 1) / self.average_count) * self.average_energy_consmption + E_t / self.average_count