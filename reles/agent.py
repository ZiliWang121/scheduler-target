import threading
import torch
import os
import logging
# from env import Env
from naf_lstm import NAF_LSTM
import mpsched
from replay_memory import ReplayMemory, Transition
from env import Env
from ounoise import OUNoise
import time
import numpy as np
from torch.autograd import Variable

# 改进1: 配置日志系统 - 减少日志混乱
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

class Online_Agent(threading.Thread):
    """Class for Online Agent thread that calls evnironment step to perform agent<->enviornment interaction as expected in reinforcement
    learning. Adjusts the split factor using the policy network of the ReLes NN after every SI until the end of the MPTCP connection
    At the start of every MPTCP connection synchronize ReLes NN (NAF+stacked LSTM) with the "offline agent" using torch.load
    Saves collected experience in replay buffer for the offline agent to use for training

    :param fd: socket file descriptor
    :type fd: int
    :param cfg: contains all the neccessary training parameter read from config.ini
    :type cfg: configParser
    :param memory: replaymemory used for adding online experience
    :type memory: class:'ReplayMemory'
    :param event: event to inform the online agent of finished MPTCP connection and no need to perform new interactions
    :type event: class:'threading.event'
    :param explore: Whether or not to use action exploration
    :explore type: boolean
    :param sender: 【修复】sender对象引用，用于获取延迟测量
    :type sender: MPTCPSender
    """

    def __init__(self, fd, cfg, memory, event, explore=True, sender=None):  # 【修复】添加sender参数
        """Constructor Method"""
        threading.Thread.__init__(self)
        self.fd = fd
        self.cfg = cfg
        self.memory = memory
        self.agent_name = cfg.get('nafcnn', 'agent')
        self.explore = explore
        self.max_flows = cfg.getint('env', 'max_num_subflows')
        # 读取目标配置
        self.target_tp = cfg.getfloat('env', 'target_tp') 
        self.target_rtt = cfg.getfloat('env', 'target_rtt') 
        self.ounoise = OUNoise(action_dimension=self.max_flows)
        self.agent = torch.load(self.agent_name)
        mpsched.persist_state(fd)
        
        # 【修复】传递sender给Env，确保延迟测量功能正常工作，并修复参数名冲突
        self.env = Env(fd=self.fd,
                       time_interval=self.cfg.getfloat('env', 'time'),  # 【修复】使用新的参数名
                       k=self.cfg.getint('env', 'k'),
                       alpha=self.cfg.getfloat('env', 'alpha'),
                       b=self.cfg.getfloat('env', 'b'),
                       c=self.cfg.getfloat('env', 'c'),
                       max_flows=self.max_flows,
                       # 添加对目标值的处理
                       target_tp=self.target_tp,
                       target_rtt=self.target_rtt,
                       sender=sender)  # 【关键修复】传递sender给Env
        self.event = event
        # 改进2: 添加模型同步相关变量
        self.step_count = 0  # 记录执行了多少步
        self.model_sync_interval = 50000  # 每50步重新加载一次模型
        self.last_model_mtime = os.path.getmtime(self.agent_name) if os.path.exists(self.agent_name) else 0 # 记录模型文件的最后修改时间
        # 改进3: 添加性能监控变量
        self.recent_rewards = []  # 存储最近的奖励值
        self.reward_window = 100  # 统计窗口大小
        
        # 【新增】：零带宽过滤相关变量
        self.throughput_threshold = 0.5  # Mbps，吞吐量阈值，可根据需要调整
        self.discarded_count = 0  # 记录丢弃的经验数量
        self.valid_experience_count = 0  # 记录有效经验数量

    def _should_reload_model(self):
        """检查是否需要重新加载模型"""
        try:
            if os.path.exists(self.agent_name):
                current_mtime = os.path.getmtime(self.agent_name)
                if current_mtime > self.last_model_mtime:
                    self.last_model_mtime = current_mtime
                    return True
        except Exception as e:
            logging.warning(f"[Online Agent] Error checking model file: {e}")
        return False

    def _reload_model(self):
        """重新加载模型"""
        try:
            self.agent = torch.load(self.agent_name)
            logging.info(f"[Online Agent] Reloaded model at step {self.step_count}")
        except Exception as e:
            logging.error(f"[Online Agent] Error reloading model: {e}")

    def _update_reward_stats(self, reward):
        """更新奖励统计"""
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window:
            self.recent_rewards.pop(0)  # 保持窗口大小

    def _log_performance(self):
        """记录性能统计（减少日志频率）"""
        if self.step_count % 20 == 0:  # 每20步记录一次，而不是每步都记录
            if self.recent_rewards:
                avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                latest_reward = self.recent_rewards[-1]
                logging.info(f"[Online Agent] Step {self.step_count}: "
                           f"Latest reward: {latest_reward:.4f}, "
                           f"Avg reward (last {len(self.recent_rewards)}): {avg_reward:.4f}")

    def run(self):
        """Override the run method from threading with the desired behaviour of the Online Agent class"""
        if True:
            self.event.wait()
            
            # 【强制验证】：在开始训练前检查延迟测量系统是否就绪
            if self.env.sender is None:
                logging.error("[Online Agent] *** CRITICAL ERROR: No sender reference for delay measurement ***")
                logging.error("[Online Agent] *** ONE-WAY DELAY SYSTEM NOT INITIALIZED ***")
                logging.error("[Online Agent] *** CANNOT PROCEED WITHOUT DELAY MEASUREMENT ***")
                return
            else:
                logging.info("[Online Agent] ✓ Delay measurement system initialized")
            
            state = self.env.reset()
            k = self.cfg.getint('env', 'k')  # k=8
            # 改进4: 减少初始化日志的频率
            logging.info(
                f"[Online Agent] Initial RTTs = {np.array(state)[self.max_flows:self.max_flows*2,-1].tolist()}"
            )
            logging.info("[Online Agent] *** STARTING TRAINING WITH MANDATORY ONE-WAY DELAY MEASUREMENT ***")
            
            state = torch.FloatTensor(state).view(-1, 1, k, 1)
            while True:
                self.step_count += 1
                # 改进5: 定期检查并重新加载模型
                if self.step_count % self.model_sync_interval == 0:
                    if self._should_reload_model():
                        self._reload_model()
                        
                # 【简化验证】：定期检查延迟测量系统状态
                if self.step_count % 20 == 0 and self.step_count > 10:  # 给更多启动时间
                    recent_delays = self.env.sender.get_recent_delays(window_ms=500)
                    if not recent_delays:
                        logging.info(f"[Online Agent] Step {self.step_count}: Still waiting for delay measurements")
                        
                start = time.time()
                if self.explore:
                    action = self.agent.select_action(state, self.ounoise)
                else:
                    action = self.agent.select_action(state)
                end = time.time()
                
                # 改进6: 减少详细日志的频率
                if self.step_count % 10 == 0:  # 每10步记录一次详细信息
                    logging.debug(f"[Online Agent] Step {self.step_count}: "
                                f"Chosen split action = {action}, "
                                f"Action compute time = {end-start:.4f}")
                state_nxt, reward, done = self.env.step((action))
                
                # 【修改开始】：添加零带宽过滤逻辑
                # 计算当前吞吐量 (Mbps)
                current_tp = (self.env.tp[0][-1] + self.env.tp[1][-1]) * 8 / (self.env.time * 1000)
                
                # 只有吞吐量大于阈值才存储经验和更新统计
                if current_tp > self.throughput_threshold:
                    # 存储有效经验
                    action_tensor = torch.FloatTensor(action)
                    mask = torch.Tensor([not done])
                    state_nxt_tensor = torch.FloatTensor(state_nxt).view(-1, 1, k, 1)
                    reward_tensor = torch.FloatTensor([float(reward)])
                    self.memory.push(state, action_tensor, mask, state_nxt_tensor, reward_tensor)
                    
                    # 更新有效经验的奖励统计
                    self._update_reward_stats(reward)
                    self._log_performance()
                    self.valid_experience_count += 1
                else:
                    # 记录丢弃的经验
                    self.discarded_count += 1
                    
                    # 定期报告过滤情况
                    if self.step_count % 50 == 0:
                        total_experiences = self.valid_experience_count + self.discarded_count
                        discard_rate = (self.discarded_count / total_experiences * 100) if total_experiences > 0 else 0
                        logging.info(f"[Online Agent] Experience filtering stats: "
                                   f"Valid: {self.valid_experience_count}, "
                                   f"Discarded: {self.discarded_count} ({discard_rate:.1f}%), "
                                   f"Current TP: {current_tp:.3f} Mbps")
                # 【修改结束】
                
                if done or not self.event.is_set():
                    logging.info(f"[Online Agent] Episode finished at step {self.step_count}")
                    break
                # 改进8: 减少状态日志频率
                if self.step_count % 20 == 0:
                    logging.debug(f"[Online Agent] Next RTTs = "
                                f"{np.array(state_nxt)[self.max_flows:self.max_flows*2,-1].tolist()}")  
                
                # 【注意】：原来的存储逻辑已经移到上面的条件判断中，这里只需要更新状态
                state = torch.FloatTensor(state_nxt).view(-1, 1, k, 1)

    def update_fd(self, fd):
        """Update the current file descriptor used in the Environment Class for reading information from subflows with socket options"""
        self.env.update_fd(fd)


class Offline_Agent(threading.Thread):
    """Class for Offline Agent that is solely resposible for training the ReLes neural network on already collected
    experience saved in the replay buffer.

    :param nn: path to pkl file to save updated NN parameters
    :type nn: string
    :param cfg: contains all the neccessary training parameter read from config.ini
    :type cfg: configParser
    :param memory: replaymemory used for reading training data to optimise the ReLes NN
    :type memory: class:'ReplayMemory'
    :param event: event indicating start/end of episode
    :type event: class:'threading.event'
    """

    def __init__(self, cfg, model, memory, event):
        """Constructor Method"""
        threading.Thread.__init__(self)
        self.memory = memory
        self.model = model
        # self.episode = cfg.getint("train","episode")
        self.batch_size = cfg.getint("train", "batch_size")
        self.event = event
        max_flows = cfg.getint("env", "max_num_subflows")
        # 改进10: 添加训练相关参数
        self.training_frequency = 1  # 每次训练5个batch而不是1个
        self.min_memory_size = self.batch_size * 2  # 最小内存要求
        self.check_interval = 60  # 每20秒检查一次，而不是等60秒
        self.save_interval = 200  # 每200次训练保存一次模型
        # 改进11: 添加训练监控变量
        self.training_step = 0
        self.recent_losses = []
        self.loss_window = 50

    def _update_loss_stats(self, loss):
        """更新损失统计"""
        self.recent_losses.append(loss)
        if len(self.recent_losses) > self.loss_window:
            self.recent_losses.pop(0)

    def _log_training_progress(self):
        """记录训练进度"""
        if self.training_step % 20 == 0:  # 每20次训练记录一次
            if self.recent_losses:
                avg_loss = sum(self.recent_losses) / len(self.recent_losses)
                latest_loss = self.recent_losses[-1]
                logging.info(f"[Offline Agent] Training step {self.training_step}: "
                           f"Latest loss: {latest_loss:.6f}, "
                           f"Avg loss (last {len(self.recent_losses)}): {avg_loss:.6f}")

    def run(self):
        """Starts the training loop for the ReLes NN"""
        # subject to change
        agent = torch.load(self.model)
        logging.info("[Offline Agent] Starting offline training loop")
        while True:
            # 改进13: 改进训练触发机制 - 更频繁的检查
            self.event.wait(timeout=self.check_interval)  # 每20秒检查而不是60秒
            # 改进17: 检查是否需要退出
            if not self.event.is_set():
                logging.info("[Offline Agent] Event cleared, preparing to exit")
                try:
                    torch.save(agent, self.model)
                    logging.info("[Offline Agent] Final model saved")
                except Exception as e:
                    logging.error(f"[Offline Agent] Error saving final model: {e}")
                break
            if len(self.memory) > self.batch_size:
                memory_size = len(self.memory)
                # 每50次训练记录一次内存状态
                if self.training_step % 50 == 0:
                    logging.info(f"[Offline Agent] Memory size = {memory_size}, "
                                f"Training step = {self.training_step}")
                for __ in range(self.training_frequency):
                    transitions = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))
                    # 更新网络参数
                    loss, _ = agent.update_parameters(batch)
                    self.training_step += 1
                    # 更新损失统计
                    self._update_loss_stats(loss)
                    self._log_training_progress()
                # 改进16: 定期保存模型，而不只是在结束时保存
                if self.training_step % self.save_interval == 0:
                        try:
                            torch.save(agent, self.model)
                            logging.info(f"[Offline Agent] Model saved at training step {self.training_step}")
                        except Exception as e:
                            logging.error(f"[Offline Agent] Error saving model: {e}")