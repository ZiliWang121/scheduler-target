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
    """
    def __init__(self,fd,time,k,alpha,b,c,max_flows,target_tp,target_rtt):
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

    def get_targets(self):
        """获取当前目标值"""
        return self.static_target_tp, self.static_target_rtt

    def adjust(self,state):
        """Converts the raw observations collected with mpsched socket api into appropriate values for state information and reward
        calculation
        
        :param state: Raw observations from socket api with mpsched extension
        :type state: list
        :return: State parameters
        :rtype: list
        """
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
        """Calculates the reward of the last SI using the ReLes reward function which consideres multiple QoS parameters
        After making measruements of path parameters with mpsched call adjust to apply changes to the Environments' state variables
        that are used for the reward calculation
        
        :return: Reward value
        :type: float
        """
        # 获取目标值
        target_tp, target_rtt = self.get_targets()
        V_throughput = self.tp[0][self.k-1] + self.tp[1][self.k-1]  # KB per SI 0.2s
        #rewards = ((self.tp[0][self.k-1])+(self.tp[1][self.k-1]))
        if V_throughput>0:
            V_RTT = (self.tp[0][self.k-1] * self.rtt[0][self.k-1] + 
                 self.tp[1][self.k-1] * self.rtt[1][self.k-1]) / V_throughput
        else:
            V_RTT = 0
        V_throughput =  V_throughput * 8 / (self.time * 1000)  # Mbps
        # V_loss = Σv_t,i (总重传包数)
        V_loss = self.in_flight[0][self.k-1] + self.in_flight[1][self.k-1]
        # 最终奖励
        reward = - abs(target_tp - V_throughput)
        print(f"[Env.reward] TP={V_throughput:.2f} Mbps, reward={reward:.3f}")
        return reward  # ← 返回正确计算的reward
        
    def reset(self):
        #在MPTCP连接开始时初始化LSTM需要的历史数据
        """Initialization of the Environment variables with the first k measurments where k is the number of past timesteps used in
        the stacked LSTM part of the NAF Q-network
        
        :return: State parameters
        :rtype: list
        """
        self.last = mpsched.get_sub_info(self.fd)
        #record k measurements
        for i in range(self.k):
            subs=mpsched.get_sub_info(self.fd)
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
        
        splits = [10, 90]
        A = [self.fd]
        SCALE   = 100                           # 精度 1 %
        """
        for k in range(self.max_num_flows):
            #用softmax后已经不需要了：ratio   = max(0.0, (action[0][k]+1)/2)   # 连续 0‥1
            weight  = int(torch.round(action[0][k] * SCALE))     # 0‥100
            splits.append(weight)
        """
        A = list(np.concatenate((A,splits)))
        print(f"[Env.step] Applied splits = {splits}")
        
        mpsched.set_seg(A)
        
        
        time.sleep(self.time)
        state_nxt = mpsched.get_sub_info(self.fd)
        print(f"[Env.step] New raw subflow state = {state_nxt}")
        done = False
        if not state_nxt:
            done = True
        state_nxt = self.adjust(state_nxt)
        reward = (self.reward())
        
        return state_nxt,reward,done
        
    
    
        
        
        
        
