import time
import mpsched
import numpy as np
import torch

class Env():
    """Evnironment class for agent interaction. 
    
    :param fd: socket file descriptor
    :type fd: int
    :param time_interval: Stater Interval (SI). usually 3~4 RTTs  # 【修复】重命名参数
    :type time_interval: int
    :param k: Number of past timesteps used in the stacked LSTM
    :type k: int
    :param alpha: first parameter of reward function to scale BDP (reduce bufferbloat|min->favors fast paths)
    :type alpha: float
    :param beta: second parameter of reward function to scale number of loss packets (reflects network congetsion|min->favors less congested paths)
    :type beta: float
    :param sender: 【修复】sender对象引用，用于获取延迟测量
    :type sender: MPTCPSender
    """
    def __init__(self,fd,time_interval,k,alpha,b,c,max_flows,target_tp,target_rtt,sender=None):  # 【修复】重命名time参数避免冲突
        """Constructor method
        """
        self.fd = fd
        self.time = time_interval  # 【修复】使用重命名的参数
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
        
        # 【修复】：延迟测量相关
        self.sender = sender  # 对sender的引用，用于获取延迟测量
        self.delay_weight = 0.3  # 延迟在reward中的权重，可根据需要调整
        
        # 【新增】：延迟测量统计
        self.delay_measurement_count = 0  # 延迟测量计数
        self.last_delay_check_time = time.time()  # 上次检查延迟的时间

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
        【关键修复】：改进reward函数，集成真实的one-way delay测量
        
        修复要点：
        1. 基于meta层面的mptcpi_bytes_acked计算真实application throughput
        2. 使用PING测量的真实端到端单向延迟（one-way delay）
        3. 更好的fallback机制和错误处理
        4. 详细的调试信息
        
        原逻辑：基于子流segments out的瞬时发送吞吐量 + 基于子流加权RTT
        新逻辑：基于meta层面确认字节数的确认吞吐量 + 基于PING的真实端到端单向延迟
        
        :return: Reward value
        :type: float
        """
        # 获取目标值
        target_tp, target_rtt = self.get_targets()
        
        # 【关键修复1】：改用meta层面的确认吞吐量计算
        if self.last_bytes_acked > 0 and self.current_bytes_acked >= self.last_bytes_acked:
            # 计算在这个SI期间确认的字节数增量
            bytes_acked_delta = self.current_bytes_acked - self.last_bytes_acked
            # 转换为Mbps：字节 -> 比特 -> Mbps
            V_throughput = (bytes_acked_delta * 8) / (self.time * 1000 * 1000)  # Mbps
        else:
            # 如果是第一次测量或者数据异常，回退到原有计算方式
            V_throughput_segments = self.tp[0][self.k-1] + self.tp[1][self.k-1]  # KB per SI
            V_throughput = V_throughput_segments * 8 / (self.time * 1000)  # Mbps
            if self.meta_measurement_count % 10 == 0:  # 偶尔提示回退
                print(f"[Env.reward] Using fallback throughput calculation")
        
        # 【关键修复2】：强制使用PING测量的真实端到端单向延迟（one-way delay）
        V_RTT = self._calculate_ping_delay()
        if V_RTT is None:
            # 【强制要求】：如果没有one-way delay数据，报错并使用明显错误值
            print(f"[Env.reward] *** ERROR: NO ONE-WAY DELAY DATA AVAILABLE ***")
            print(f"[Env.reward] *** THIS SHOULD NOT HAPPEN - DELAY MEASUREMENT SYSTEM FAILED ***")
            print(f"[Env.reward] *** USING INVALID DELAY VALUE -999999.0ms ***")
            V_RTT = -999999.0  # 使用明显的错误值，这样reward会很差，强制系统修复延迟测量
        else:
            # 使用PING测量的延迟（已经是ms单位）
            if self.delay_measurement_count % 5 == 0:  # 每5次测量打印一次
                print(f"[Env.reward] ✓ USING ONE-WAY DELAY: {V_RTT:.1f}ms")
        
        # 【保持原有loss计算】：基于子流重传数据
        V_loss = self.in_flight[0][self.k-1] + self.in_flight[1][self.k-1]
        
        # 【修复reward函数】：综合考虑吞吐量和延迟
        # 将延迟从ms转换为s以匹配目标单位
        V_RTT_seconds = V_RTT / 1000.0
        
        throughput_reward = -abs(target_tp - V_throughput)
        delay_reward = -abs(target_rtt - V_RTT_seconds)
        
        # 加权组合：吞吐量占主导，延迟作为补充
        reward = (1 - self.delay_weight) * throughput_reward + self.delay_weight * delay_reward
        
        # 【增强的调试信息】：明确显示新旧计算方式的对比和延迟信息
        if self.meta_measurement_count % 10 == 0:  # 每10次计算显示一次详细对比
            segments_tp = (self.tp[0][self.k-1] + self.tp[1][self.k-1]) * 8 / (self.time * 1000)
            print(f"[Env.reward] META_TP={V_throughput:.3f} Mbps, "
                  f"SEGMENTS_TP={segments_tp:.3f} Mbps, "
                  f"DELAY={V_RTT:.1f}ms, "
                  f"TARGET_TP={target_tp:.3f}, TARGET_RTT={target_rtt*1000:.1f}ms, "
                  f"reward={reward:.3f}")
        else:
            print(f"[Env.reward] TP={V_throughput:.2f} Mbps, DELAY={V_RTT:.1f}ms, reward={reward:.3f}")
            
        return reward
    
    def _calculate_ping_delay(self):
        """
        【强制One-Way Delay】：从sender获取最近的one-way delay测量并计算平均值
        
        这个函数必须返回PING测量的延迟，不允许fallback到RTT！
        
        修复要点：
        1. 更严格的数据验证
        2. 更好的异常值处理
        3. 详细的调试信息
        4. 强制要求有PING数据，否则返回None
        
        Returns:
            float or None: 平均单向延迟(ms)，如果没有可用数据则返回None
        """
        if self.sender is None:
            if self.delay_measurement_count % 20 == 0:  # 减少日志频率
                print("[Env._calculate_ping_delay] *** CRITICAL: No sender reference - PING delay system not initialized ***")
            return None
            
        try:
            # 【修复】：获取最近150ms内的延迟测量（与SI时间窗口对齐）
            recent_delays = self.sender.get_recent_delays(window_ms=150)
            
            current_time = time.time()
            time_since_last_check = current_time - self.last_delay_check_time
            
            if not recent_delays:
                # 【强制要求】：必须有PING延迟数据
                if self.delay_measurement_count % 5 == 0:  # 增加日志频率以便调试
                    print(f"[Env._calculate_ping_delay] *** WARNING: NO PING DELAY DATA in 150ms window ***")
                    print(f"[Env._calculate_ping_delay] *** Time since last check: {time_since_last_check:.3f}s ***")
                    print(f"[Env._calculate_ping_delay] *** This indicates PING system is not working properly ***")
                return None
                
            # 【修复】：更严格的数据验证和异常值处理
            valid_delays = []
            invalid_count = 0
            for delay in recent_delays:
                # 延迟合理性检查：1ms到2000ms之间
                if 1.0 <= delay <= 2000.0:
                    valid_delays.append(delay)
                else:
                    invalid_count += 1
                    if invalid_count <= 3:  # 只打印前3个无效值
                        print(f"[Env._calculate_ping_delay] Discarding invalid delay: {delay:.1f}ms")
            
            if not valid_delays:
                print(f"[Env._calculate_ping_delay] *** ERROR: All {len(recent_delays)} PING delays were invalid ***")
                print(f"[Env._calculate_ping_delay] *** Raw delays: {recent_delays[:5]}... ***")
                return None
            
            # 【修复】：计算平均延迟，如果样本足够则排除异常值
            if len(valid_delays) >= 5:
                # 如果有足够的样本，排除最大和最小值后计算平均
                sorted_delays = sorted(valid_delays)
                # 排除最大和最小的10%
                trim_count = max(1, len(sorted_delays) // 10)
                trimmed_delays = sorted_delays[trim_count:-trim_count] if trim_count > 0 else sorted_delays
                avg_delay = sum(trimmed_delays) / len(trimmed_delays)
                
                if self.delay_measurement_count % 5 == 0:
                    print(f"[Env._calculate_ping_delay] ✓ PING DELAY SUCCESS: {len(trimmed_delays)}/{len(valid_delays)} delays, "
                          f"avg={avg_delay:.1f}ms, range=[{min(trimmed_delays):.1f}, {max(trimmed_delays):.1f}]ms")
            elif len(valid_delays) >= 2:
                # 样本数较少但足够时直接平均
                avg_delay = sum(valid_delays) / len(valid_delays)
                if self.delay_measurement_count % 5 == 0:
                    print(f"[Env._calculate_ping_delay] ✓ PING DELAY SUCCESS: {len(valid_delays)} delays, avg={avg_delay:.1f}ms")
            else:
                # 样本数太少，不可靠
                print(f"[Env._calculate_ping_delay] *** WARNING: Only {len(valid_delays)} valid PING delay(s), may be unreliable ***")
                avg_delay = valid_delays[0]
                print(f"[Env._calculate_ping_delay] ✓ Using single PING delay: {avg_delay:.1f}ms")
            
            self.delay_measurement_count += 1
            self.last_delay_check_time = current_time
            return avg_delay
            
        except Exception as e:
            print(f"[Env._calculate_ping_delay] *** CRITICAL ERROR: {e} ***")
            import traceback
            traceback.print_exc()
            return None
        
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
        
        # 【新增】：重置延迟测量变量
        self.delay_measurement_count = 0
        self.last_delay_check_time = time.time()
        
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