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
    """处理MPTCP连接和单文件发送的线程"""
    def __init__(self, cfg, memory, file_to_send):  # 修改1: 改为单文件而不是文件列表
        threading.Thread.__init__(self)
        self.cfg = cfg
        self.memory = memory
        self.file_to_send = file_to_send  # 修改2: 存储单个文件名
        self.IP = cfg.get('receiver','ip')
        self.PORT = cfg.getint('receiver','port')
        self.transfer_event = Event()
        self.result = None  # 修改3: 改为单个结果而不是结果列表
        
    def run(self):
        """建立MPTCP连接，发送单个文件"""  # 修改4: 注释改为单文件
        # 创建MPTCP socket
        MPTCP_ENABLED = 42
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 添加连接超时
        
        # 启用MPTCP
        try:
            sock.setsockopt(socket.IPPROTO_TCP, MPTCP_ENABLED, 1)
        except:
            print("[Sender] Warning: Could not enable MPTCP")
        
        try:
            # 建立连接
            sock.connect((self.IP, self.PORT))
            sock.settimeout(None)  # 连接后取消超时
            fd = sock.fileno()
            mpsched.persist_state(fd)
            
            # 启动Online Agent（整个连接期间运行）
            agent = Online_Agent(fd=fd, cfg=self.cfg, memory=self.memory, 
                               event=self.transfer_event)
            agent.start()
            self.transfer_event.set()
            
            # 修改5: 发送单个文件，移除循环和sleep
            start_time = time.time()
            
            try:
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
                
                end_time = time.time()
                completion_time = end_time - start_time
                
                # 修改6: 保存单个结果
                self.result = {
                    'file': self.file_to_send,
                    'completion_time': completion_time,
                    'success': True
                }
                
                print(f"[Sender] File sent: {self.file_to_send} ({completion_time:.2f}s)")
                
            except Exception as e:
                print(f"[Sender] Error sending {self.file_to_send}: {e}")
                # 修改7: 错误时也保存单个结果
                self.result = {
                    'file': self.file_to_send,
                    'completion_time': 0,
                    'success': False
                }
            
        except socket.timeout:
            print("[Sender] Connection timeout")
            # 修改8: 超时时保存单个结果
            self.result = {
                'file': self.file_to_send,
                'completion_time': 0,
                'success': False
            }
        except Exception as e:
            print(f"[Sender] Connection error: {e}")
            # 修改9: 连接错误时保存单个结果
            self.result = {
                'file': self.file_to_send,
                'completion_time': 0,
                'success': False
            }
        finally:
            self.transfer_event.clear()
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
    FILE = cfg.get('file','file')  # 保持原有从config读取文件的逻辑
    FILES = ["64kb.dat","2mb.dat","8mb.dat","64mb.dat"]
    
    # 创建训练专用的event（永不清除）
    training_event = Event()
    training_event.set()
    CONTINUE_TRAIN = 1
    num_iterations = 150
    scenario = "default"
    # 修改10: 移除batch_size参数，改为单文件模式
    
    # 解析命令行参数
    if len(argv) >= 1:
        CONTINUE_TRAIN = int(argv[0])
    if len(argv) >= 2:
        scenario = argv[1]
    if len(argv) >= 3:
        FILE = argv[2]
    if len(argv) >= 4:
        num_iterations = int(argv[3])
    # 修改11: 移除batch_size参数解析
        
    now = datetime.now().replace(microsecond=0)
    start_train = now.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[Sender] Starting with args CONTINUE_TRAIN={CONTINUE_TRAIN}, scenario='{scenario}'")
    print(f"[Sender] FILE={FILE}, iterations={num_iterations}")  # 修改12: 移除batch_size显示
    
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
        action_space=MAX_NUM_FLOWS)
        torch.save(agent,AGENT_FILE)

    # 初始化训练线程变量
    off_agent = None
    
    # 用于保存性能指标
    performance_metrics = []
    np.random.seed(42)
    
    try:
        # 修改13: 主循环改为逐个发送文件而不是批量发送
        for iteration in range(1, num_iterations + 1):
            
            # 选择要发送的文件（保持原有逻辑）
            if FILE == "random" and num_iterations > 150:
                FILE2 = np.random.choice(FILES,p=[0, 0.9, 0, 0.1])
            elif FILE == "random" and num_iterations == 150:
                FILE2 = np.random.choice(FILES,p=[0.3,0.35,0.3,0.05])
            else:
                FILE2 = FILE
            
            print(f"\n[Sender] Iteration {iteration}/{num_iterations}: Sending {FILE2}")
            
            # 修改14: 创建发送线程（每次发送一个文件）
            sender = MPTCPSender(cfg, memory, FILE2)
            sender.start()
            sender.join(timeout=60)  # 60秒超时
            
            if sender.is_alive():
                print(f"[Sender] Iteration {iteration} timeout, skipping")
                continue
            
            # 修改15: 处理单个结果而不是结果列表
            result = sender.result
            if result and result['success']:
                completion_time = result['completion_time']
                file_name = result['file']
                
                # 计算性能指标（保持原有逻辑）
                if file_name.find("kb") != -1:
                    file_size = int(re.findall(r'\d+',file_name)[0])/1000
                else: 
                    file_size = int(re.findall(r'\d+',file_name)[0])
                    
                throughput = file_size/completion_time if completion_time > 0 else 0
                
                # 保存性能指标（保持原有逻辑）
                if iteration >= 30:
                    performance_metrics.append({
                        "iteration": iteration,
                        "file": file_name,
                        "completion time": completion_time,
                        "throughput": throughput,
                        "file_size_MB": file_size
                    })
                
                print(f"[Sender] Iteration {iteration}: {file_name} - {throughput:.2f} MB/s")
            else:
                print(f"[Sender] Iteration {iteration}: {FILE2} - FAILED")
            
            # 检查是否需要启动离线训练（保持原有逻辑）
            if len(memory) > BATCH_SIZE:
                if off_agent is None or not off_agent.is_alive():
                    off_agent = Offline_Agent(cfg=cfg, model=AGENT_FILE, 
                                             memory=memory, event=training_event)
                    off_agent.daemon = True
                    off_agent.start()
                    print(f"[Sender] Training started/restarted after iteration {iteration}")
                    print(f"[Sender] Memory size: {len(memory)}")
            
            # 修改16: 迭代间间隔改为2秒，给系统时间清理资源
            time.sleep(2)
            
        # 保存性能指标（保持原有逻辑）
        if performance_metrics:
            df = pd.DataFrame(performance_metrics)
            df.to_csv("sender_metrics.csv", index=False)
            print("[Sender] Performance metrics saved to sender_metrics.csv")
            
    except (KeyboardInterrupt, SystemExit):
        print("\n[Sender] Shutting down...")
        training_event.clear()
        if off_agent and off_agent.is_alive():
            print("[Sender] Waiting for training thread to finish...")
            off_agent.join(timeout=5)
    finally:
        # 保存replay memory（保持原有逻辑）
        with open(MEMORY_FILE,'wb') as f:
            pickle.dump(memory,f)
            f.close()
        print("[Sender] Memory saved")

if __name__ == '__main__':
    main(sys.argv[1:])