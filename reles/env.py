import time
import mpsched
import numpy as np
import torch

class Env():
    """Evnironment class for agent interaction. 
    
    :param fd: socket file descriptor
    :type fd: int
    :param time: Stater Interval (SI). usually 3~4 RTTs
    :type time: int
    :param k: Number of past timesteps used in the stacked LSTM
    :type k: int
    :param alpha: first parameter of reward function to scale BDP (reduce bufferbloat|min->favors fast paths)
    :type alpha: float
    :param beta: second parameter of reward function to scale number of loss packets (reflects network congetsion|min->favors less congested paths)
    :type beta: float
    :param sender: 【新增】MPTCPSender引用，用于获取延迟测量数据
    :type sender: MPTCPSender or None
    """
    def __init__(self,fd,time,k,alpha,b,c,max_flows,target_tp,target_rtt,sender=None):  # 【修改】添加sender参数
        """Constructor method
        """
        self.fd = fd
        self.time = time
        self.k = k
        self.alpha = alpha
        self.b = b
        self.c = c
        self.num_segments = 10
        self.max_num_flows = max_flows
        self.static_target_tp = target_tp
        self.static_target_rtt = target_rtt
        # 初始化目标序列
        self.target_tp = []      # 全局吞吐量目标序列
        self.target_rtt = []     # 全局RTT目标序列
        self.last = [[0]*5 for _ in range(self.max_num_flows)]
        self.tp = [[] for _ in range(self.max_num_flows)]        #num of segments out (MSS=1440B)
        self.rtt = [[] for _ in range(self.max_num_flows)]        #snapshot of rtt
        self.dRtt = [[] for _ in range(self.max_num_flows)]
        self.cwnd = [[] for _ in range(self.max_num_flows)]        #cwnd sender
        self.rr = [[] for _ in range(self.max_num_flows)]        #number of unacked packets (in flight)
        self.in_flight = [[] for _ in range(self.max_num_flows)]    #number of TOTAL retransmissions
        
        # 【新增】：用于跟踪meta层面的确认字节数，计算真实吞吐量
        self.last_bytes_acked = 0    # 上一次的meta.mptcpi_bytes_acked值
        self.current_bytes_acked = 0 # 当前的meta.mptcpi_bytes_acked值
        
        # 【新增】：用于调试和验证的计数器
        self.meta_measurement_count = 0  # meta测量次数计数
        
        # 【新增】：延迟测量相关
        self.sender = sender  # MPTCPSender引用，用于获取延迟数据
        
        # 【新增】：延迟reward的权重配置
        self.delay_weight = 0.1  # 延迟在reward中的权重，可调节

    def get_targets(self):
        """获取当前目标值"""
        return self.static_target_tp, self.static_target_rtt

    def _extract_meta_info(self, raw_state):
        """
        从get_sub_info返回的原始数据中提取meta信息
        
        :param raw_state: mpsched.get_sub_info()返回的原始列表
        :return: (subflow_data, bytes_acked) 元组
        """
        if not raw_state:
            return [], None
            
        # 检查是否包含meta信息：最后一个元素且长度为1（只有bytes_acked）
        if len(raw_state) > 0 and isinstance(raw_state[-1], list) and len(raw_state[-1]) == 1:
            # 分离子流数据和meta数据
            subflow_data = raw_state[:-1]  # 除了最后一个元素的所有子流数据
            bytes_acked = raw_state[-1][0] # 最后一个元素的第一个值是bytes_acked
            return subflow_data, bytes_acked
        else:
            # 兼容旧版本：没有meta信息的情况
            return raw_state, None

    def _update_meta_tracking(self, bytes_acked):
        """
        更新meta层面的字节确认跟踪
        
        :param bytes_acked: meta.mptcpi_bytes_acked值
        """
        if bytes_acked is not None:
            # 更新字节确认跟踪
            self.last_bytes_acked = self.current_bytes_acked
            self.current_bytes_acked = bytes_acked
            self.meta_measurement_count += 1

    def adjust(self,state):
        """Converts the raw observations collected with mpsched socket api into appropriate values for state information and reward
        calculation
        
        :param state: Raw observations from socket api with mpsched extension
        :type state: list
        :return: State parameters
        :rtype: list
        """
        # 【新增】：提取并处理meta信息
        subflow_state, bytes_acked = self._extract_meta_info(state)
        
        # 【新增】：更新meta层面的跟踪信息（用于reward计算）
        self._update_meta_tracking(bytes_acked)
        
        # 【保持原有逻辑】：处理子流状态信息（用于state构建）
        state = subflow_state  # 使用分离后的子流数据继续原有处理逻辑
        
        # 1. 首先处理目标序列（全局的，只处理一次）
        if len(self.target_tp) == self.k:
            self.target_tp.pop(0)
            self.target_rtt.pop(0)
        # 2. 然后处理每个子流的序列
        for i in range(len(state)): #in range 2 if len sate < 2 etc.
            if len(self.tp[i]) == self.k:
                self.tp[i].pop(0)
                self.rtt[i].pop(0)
                self.dRtt[i].pop(0)
                self.cwnd[i].pop(0)
                self.rr[i].pop(0)
                self.in_flight[i].pop(0)
            if len(self.last)< self.max_num_flows:
                #if not all subflows appeared yet set rest to 0
                for _ in range(self.max_num_flows-len(self.last)):
                    self.last.append([0,0,0,0,0])
            if len(state) < self.max_num_flows:
                for _ in range(self.max_num_flows-len(state)):
                    state.append([0,0,0,0,0])
            self.tp[i].append(np.abs(state[i][0]-self.last[i][0])*1.44)
            #self.tp[i].append(np.abs(state[i][0]-self.last[i][0])*1.5)
            self.rtt[i].append((state[i][1])/1000)
            self.dRtt[i].append(state[i][1]-self.last[i][1])
            self.cwnd[i].append((state[i][2]+self.last[i][2])/2)
            #self.cwnd[i].append(state[i][2])
            self.rr[i].append(np.abs(state[i][3]-self.last[i][3]))
            self.in_flight[i].append(np.abs(state[i][4]-self.last[i][4]))#look at wording in reles paper
        self.last = state
        # 获取目标值并添加到全局目标序列
        target_tp, target_rtt = self.get_targets()
        self.target_tp.append(target_tp)
        self.target_rtt.append(target_rtt)
        return [self.tp[0],self.tp[1],self.rtt[0],self.rtt[1],self.cwnd[0],self.cwnd[1],self.rr[0],self.rr[1],
        self.in_flight[0],self.in_flight[1],self.target_tp,self.target_rtt]
        
    def reward(self):
        """
        【重要修改】：基于meta层面的mptcpi_bytes_acked计算真实application throughput
        【新增功能】：加入基于实测延迟的reward计算
        
        原逻辑：只考虑吞吐量 reward = -abs(target_tp - actual_tp)
        新逻辑：同时考虑吞吐量和延迟 reward = throughput_reward + delay_reward
        
        :return: Reward value
        :type: float
        """
        # 获取目标值
        target_tp, target_rtt = self.get_targets()
        
        # 【保持原有】：计算吞吐量
        if self.last_bytes_acked > 0 and self.current_bytes_acked >= self.last_bytes_acked:
            # 计算在这个SI期间确认的字节数增量
            bytes_acked_delta = self.current_bytes_acked - self.last_bytes_acked
            # 转换为Mbps：字节 -> 比特 -> Mbps
            V_throughput = (bytes_acked_delta * 8) / (self.time * 1000 * 1000)  # Mbps
        else:
            # 如果是第一次测量或者数据异常，回退到原有计算方式
            V_throughput_segments = self.tp[0][self.k-1] + self.tp[1][self.k-1]  # KB per SI
            V_throughput = V_throughput_segments * 8 / (self.time * 1000)  # Mbps
            print(f"[Env.reward] Using fallback calculation (segments-based)")
        
        # 【新增】：计算实测延迟
        measured_rtt = target_rtt  # 默认值：使用目标延迟
        delay_sample_count = 0
        
        if self.sender:
            # 从sender获取最近150ms的延迟测量
            recent_delays = self.sender.get_recent_delays(window_ms=150)
            if len(recent_delays) >= 2:  # 至少需要2个测量点才可靠
                measured_rtt = sum(recent_delays) / len(recent_delays)
                delay_sample_count = len(recent_delays)
            # 如果没有足够的延迟数据，就使用目标值（保持原有行为）
        
        # 【修改】：计算综合reward = 吞吐量reward + 延迟reward
        throughput_penalty = abs(target_tp - V_throughput)  # 吞吐量偏差
        delay_penalty = abs(target_rtt - measured_rtt)      # 延迟偏差
        
        # 综合reward：两个部分都是惩罚项（越小越好），所以都是负值
        throughput_reward = -throughput_penalty
        delay_reward = -delay_penalty * self.delay_weight  # 延迟权重较小
        
        total_reward = throughput_reward + delay_reward
        
        # 【增强的调试信息】：显示新的reward计算过程
        if self.meta_measurement_count % 5 == 0:  # 每5次计算显示一次详细信息
            segments_tp = (self.tp[0][self.k-1] + self.tp[1][self.k-1]) * 8 / (self.time * 1000)
            print(f"[Env.reward] TP={V_throughput:.2f}Mbps(target={target_tp:.2f}), "
                  f"RTT={measured_rtt:.1f}ms(target={target_rtt:.1f}, samples={delay_sample_count}), "
                  f"reward={total_reward:.3f}(tp:{throughput_reward:.3f}+delay:{delay_reward:.3f})")
        else:
            print(f"[Env.reward] TP={V_throughput:.2f}Mbps, RTT={measured_rtt:.1f}ms, reward={total_reward:.3f}")
            
        return total_reward
        
    def reset(self):
        #在MPTCP连接开始时初始化LSTM需要的历史数据
        """Initialization of the Environment variables with the first k measurments where k is the number of past timesteps used in
        the stacked LSTM part of the NAF Q-network
        
        :return: State parameters
        :rtype: list
        """
        # 【新增】：重置meta跟踪变量
        self.last_bytes_acked = 0
        self.current_bytes_acked = 0
        self.meta_measurement_count = 0
        
        raw_state = mpsched.get_sub_info(self.fd)
        subflow_state, bytes_acked = self._extract_meta_info(raw_state)
        
        # 【新增】：初始化meta跟踪（但不用于reward计算，因为还没有增量）
        if bytes_acked:
            self.current_bytes_acked = bytes_acked
            self.last_bytes_acked = self.current_bytes_acked  # 初始状态：上一次=当前次
            print(f"[Env.reset] Initial meta bytes_acked = {self.current_bytes_acked}")
        
        self.last = subflow_state
        
        #record k measurements
        for i in range(self.k):
            raw_subs = mpsched.get_sub_info(self.fd)
            subs, bytes_acked = self._extract_meta_info(raw_subs)
            
            # 【新增】：在reset期间也要更新meta跟踪
            self._update_meta_tracking(bytes_acked)
            
            for j in range(self.max_num_flows):
                if len(self.tp[j]) == self.k:
                    self.tp[j].pop(0)
                    self.rtt[j].pop(0)
                    self.dRtt[j].pop(0)
                    self.cwnd[j].pop(0)
                    self.rr[j].pop(0)
                    self.in_flight[j].pop(0)
                if len(self.last)<(self.max_num_flows):
                    for _ in range(self.max_num_flows-len(self.last)):                
                        self.last.append([0,0,0,0,0])
                if len(subs)<self.max_num_flows:
                    for _ in range(self.max_num_flows-len(subs)):
                        subs.append([0,0,0,0,0])
                self.tp[j].append(np.abs(subs[j][0]-self.last[j][0])*1.44)
                self.rtt[j].append((subs[j][1]/1000))
                self.dRtt[j].append(np.abs(subs[j][1]-self.last[j][1]))
                self.cwnd[j].append((subs[j][2]+self.last[j][2])/2)
                self.rr[j].append(np.abs(subs[j][3]-self.last[j][3]))
                self.in_flight[j].append(np.abs(subs[j][4]-self.last[j][4]))
            self.last = subs
            time.sleep((self.time)/10) 
            
        # 初始化目标序列：填充k个相同的目标值
        target_tp, target_rtt = self.get_targets()
        for _ in range(self.k):
            if len(self.target_tp) == self.k:
                self.target_tp.pop(0)
                self.target_rtt.pop(0)
            self.target_tp.append(target_tp)
            self.target_rtt.append(target_rtt)
        return [self.tp[0],self.tp[1],self.rtt[0],self.rtt[1],self.cwnd[0],self.cwnd[1],self.rr[0],self.rr[1],
        self.in_flight[0],self.in_flight[1],self.target_tp,self.target_rtt]
        
    def update_fd(self,fd):
        self.fd = fd
    
    def step(self,action):
        """Performs all neccessary actions to transition the Environment from SI t into t+1.
        Actions include among other things: 
        -setting the split factor for the kernel scheduler using socket api with mpsched extension
        -calculated the reward of the action of state t using reward method
        -wait SI until the begin of the next state t+1
        -take measurement of the new path characteristics after taking action t using socket api with mpsched extension
        -adjust the current environment variables using adjust method
        
        :param action: split factor derived using the current policy network of the ReLes NAF NN 
        :type action: list
        :return: state observation of the next state t+1,reward value and boolean indication whether bulk transfer is over 
        :rtype: list,float,boolean
        """
        splits = []
        A = [self.fd]
        SCALE   = 100                           # 精度 1 %
        #" " "
        for k in range(self.max_num_flows):
            #用softmax后已经不需要了：ratio   = max(0.0, (action[0][k]+1)/2)   # 连续 0‥1
            weight  = int(torch.round(action[0][k] * SCALE))     # 0‥100
            splits.append(weight)
        # " " "
        #splits = [20, 80]
        A = list(np.concatenate((A,splits)))
        print(f"[Env.step] Applied splits = {splits}")
        # 只加这两行调试
        #print(f"[DEBUG] Before set_seg: {A}")
        mpsched.set_seg(A)
        #print(f"[DEBUG] set_seg returned: {result}")
        #mpsched.set_seg(A)
        
        time.sleep(self.time)
        raw_state_nxt = mpsched.get_sub_info(self.fd)
        subflow_state_nxt, _ = self._extract_meta_info(raw_state_nxt)
        
        print(f"[Env.step] New raw subflow state = {subflow_state_nxt}")
        done = False
        if not subflow_state_nxt:
            done = True
            
        # 【保持原有接口】：adjust()和reward()的调用方式不变，但内部逻辑已更新
        state_nxt = self.adjust(raw_state_nxt)  # 传入完整原始数据，let adjust()处理meta分离
        reward = (self.reward())
        
        return state_nxt,reward,done