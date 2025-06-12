#!/usr/bin/python3

import sys
import os
from os import path
import time
import threading
import pickle
from threading import Event
import socketserver
import numpy as np
import socket
import mpsched
import torch
from configparser import ConfigParser
from replay_memory import ReplayMemory
from agent import Online_Agent, Offline_Agent
from naf_lstm import NAF_LSTM
from gym import spaces
import multiprocessing
from datetime import datetime
import shutil
import pandas as pd
import re
import random

#structure and modulisation based on github.com/gaogogo/Experiment

class MPTCPSender(threading.Thread):
    """处理单个MPTCP连接的发送线程"""
    def __init__(self, cfg, memory, file_to_send):
        threading.Thread.__init__(self)
        self.cfg = cfg
        self.memory = memory
        self.file_to_send = file_to_send
        self.IP = cfg.get('receiver','ip')
        self.PORT = cfg.getint('receiver','port')
        # 创建传输专用的event
        self.transfer_event = Event()
        
    def run(self):
        """建立MPTCP连接并发送文件"""
        # 创建MPTCP socket
        MPTCP_ENABLED = 42  # 根据您的内核版本可能需要调整
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 启用MPTCP（如果系统支持）
        try:
            sock.setsockopt(socket.IPPROTO_TCP, MPTCP_ENABLED, 1)
        except:
            print("[Sender] Warning: Could not enable MPTCP")
        
        try:
            # 连接到接收端
            sock.connect((self.IP, self.PORT))
            fd = sock.fileno()
            mpsched.persist_state(fd)
            
            # 启动Online Agent
            # 使用独立的event控制
            agent = Online_Agent(fd=fd, cfg=self.cfg, memory=self.memory, 
                               event=self.transfer_event)
            # 启动Online Agent
            # agent = Online_Agent(fd=fd, cfg=self.cfg, memory=self.memory, event=self.event)
            agent.start()
            self.transfer_event.set()  # 启动智能体
            
            # 发送文件名
            filename_msg = f"FILE:{self.file_to_send}\n".encode('utf-8')
            sock.send(filename_msg)
            
            # 发送文件内容
            with open(self.file_to_send, 'rb') as f:
                while True:
                    data = f.read(4096)
                    if not data:
                        break
                    sock.sendall(data)
            
            # 等待一小段时间确保数据传输完成
            time.sleep(0.5)
            
        except Exception as e:
            print(f"[Sender] Error: {e}")
        finally:
            self.transfer_event.clear()  # 只影响当前传输
            sock.close()

def main(argv):
    cfg = ConfigParser()
    cfg.read('config.ini')
    
    MEMORY_FILE = cfg.get('replaymemory','memory')
    AGENT_FILE = cfg.get('nafcnn','agent')
    INTERVAL = cfg.getint('train','interval')
    EPISODE = cfg.getint('train','episode')
    BATCH_SIZE = cfg.getint('train','batch_size')
    MAX_NUM_FLOWS = cfg.getint("env",'max_num_subflows')
    FILE = cfg.get('file','file')
    FILES = ["64kb.dat","2mb.dat","8mb.dat","64mb.dat"]
    
    # 创建训练专用的event（永不清除）
    training_event = Event()
    training_event.set()  # 立即设置，保持训练活跃
    CONTINUE_TRAIN = 1
    num_iterations = 150
    scenario = "default"
    
    # 解析命令行参数
    if len(argv) >= 1:
        CONTINUE_TRAIN = int(argv[0])
    if len(argv) >= 2:
        scenario = argv[1]
    if len(argv) >= 3:
        FILE = argv[2]
    if len(argv) >= 4:
        num_iterations = int(argv[3])
        
    now = datetime.now().replace(microsecond=0)
    start_train = now.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[Sender] Starting with args CONTINUE_TRAIN={CONTINUE_TRAIN}, scenario='{scenario}'")
    print(f"[Sender] FILE={FILE}, iterations={num_iterations}")
    
    # 加载或创建replay memory
    if os.path.exists(MEMORY_FILE) and CONTINUE_TRAIN:
        with open(MEMORY_FILE,'rb') as f:
            try:
                memory = pickle.load(f)
                f.close()
            except EOFError:
                print("memory EOF error not saved properly")
                memory = ReplayMemory(cfg.getint('replaymemory','capacity'))
    else:
        memory = ReplayMemory(cfg.getint('replaymemory','capacity'))

    # 处理agent文件
    if CONTINUE_TRAIN != 1 and os.path.exists(AGENT_FILE):
        os.makedirs("trained_models/",exist_ok=True)
        shutil.move(AGENT_FILE,"trained_models/agent"+start_train+".pkl")
    if not os.path.exists(AGENT_FILE) or CONTINUE_TRAIN != 1:
        agent = NAF_LSTM(gamma=cfg.getfloat('nafcnn','gamma'),tau=cfg.getfloat('nafcnn','tau'),
        hidden_size=cfg.getint('nafcnn','hidden_size'),num_inputs=cfg.getint('env','k')*MAX_NUM_FLOWS*5,
        action_space=MAX_NUM_FLOWS) #5 is the size of state space (TP,RTT,CWND,unACK,retrans)
        torch.save(agent,AGENT_FILE)

    # 初始化训练线程变量
    off_agent = None
    
    # 用于保存性能指标（类似原client.py）
    performance_metrics = []
    np.random.seed(42)
    
    try:
        # 主循环：发送多个文件
        for i in range(num_iterations):
            # 选择要发送的文件（保留原client.py的逻辑）
            if FILE == "random" and num_iterations > 150:
                FILE2 = np.random.choice(FILES,p=[0, 0.9, 0, 0.1])
            elif FILE == "random" and num_iterations == 150:
                FILE2 = np.random.choice(FILES,p=[0.3,0.35,0.3,0.05])
            else:
                FILE2 = FILE
                
            print(f"\n[Sender] Iteration {i+1}/{num_iterations}, sending: {FILE2}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 创建发送线程
            # 修改：移除event参数
            sender = MPTCPSender(cfg, memory, FILE2)
            sender.start()
            sender.join()  # 等待发送完成
            
            # 记录结束时间
            end_time = time.time()
            completion_time = end_time - start_time
            
            # 计算性能指标
            if FILE2.find("kb") != -1:
                file_size = int(re.findall(r'\d+',FILE2)[0])/1000
            else: 
                file_size = int(re.findall(r'\d+',FILE2)[0])
                
            throughput = file_size/completion_time if completion_time > 0 else 0
            
            # 保存性能指标（如果迭代次数>=30，类似原client.py）
            if i >= 30:
                performance_metrics.append({
                    "iteration": i,
                    "file": FILE2,
                    "completion time": completion_time,
                    "throughput": throughput,
                    "file_size_MB": file_size
                })
            
            # 检查是否需要启动离线训练
            # 修复：正确的训练线程管理
            if len(memory) > BATCH_SIZE:
                if off_agent is None or not off_agent.is_alive():
                    # 创建新的训练线程
                    off_agent = Offline_Agent(cfg=cfg, model=AGENT_FILE, 
                                             memory=memory, event=training_event)
                    off_agent.daemon = True
                    off_agent.start()
                    print(f"[Sender] Training started/restarted at iteration {i+1}")
                    print(f"[Sender] Memory size: {len(memory)}")
                
            # 短暂休息（类似原client.py）
            time.sleep(0.25)
            
        # 保存性能指标
        if performance_metrics:
            df = pd.DataFrame(performance_metrics)
            df.to_csv("sender_metrics.csv", index=False)
            print("[Sender] Performance metrics saved to sender_metrics.csv")
            
    except (KeyboardInterrupt, SystemExit):
        print("\n[Sender] Shutting down...")
        training_event.clear()  # 停止训练
        if off_agent and off_agent.is_alive():
            print("[Sender] Waiting for training thread to finish...")
            off_agent.join(timeout=5)
    finally:
        # 保存replay memory
        with open(MEMORY_FILE,'wb') as f:
            pickle.dump(memory,f)
            f.close()
        print("[Sender] Memory saved")

if __name__ == '__main__':
    main(sys.argv[1:])
