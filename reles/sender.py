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
import struct  # 【新增】用于打包/解包二进制数据

#structure and modulisation based on github.com/gaogogo/Experiment

class MPTCPSender(threading.Thread):
    """处理MPTCP连接和多文件发送的线程"""
    def __init__(self, cfg, memory, file_list):
        threading.Thread.__init__(self)
        self.cfg = cfg
        self.memory = memory
        self.file_list = file_list  # 改为文件列表
        self.IP = cfg.get('receiver','ip')
        self.PORT = cfg.getint('receiver','port')
        self.transfer_event = Event()
        self.results = []  # 存储每个文件的传输结果
        
        # 【新增】延迟测量相关属性
        self.delay_measurements = []  # 存储格式: [(delay_ms, received_timestamp), ...]
        self.delay_lock = threading.Lock()  # 线程安全锁
        self.probe_sequence = 0  # 探测包序号
        self.probe_interval = 0.05  # 50ms发送一次探测包
        self.pending_probes = {}  # 记录已发送但未收到回复的探测包: {seq: sent_timestamp}
        
    # 【新增】获取最近延迟数据的接口，供Env调用
    def get_recent_delays(self, window_ms=150):
        """获取最近window_ms毫秒内收到的延迟测量
        
        Args:
            window_ms: 时间窗口，默认150ms（避免跨action的数据污染）
            
        Returns:
            list: 最近的延迟测量值列表 [delay1_ms, delay2_ms, ...]
        """
        current_time = time.time()
        cutoff_time = current_time - (window_ms / 1000.0)
        
        with self.delay_lock:
            recent_delays = [
                delay_ms for delay_ms, received_time in self.delay_measurements
                if received_time >= cutoff_time
            ]
            
        return recent_delays
    
    # 【新增】延迟探测工作线程
    def delay_probe_worker(self, sock):
        """每50ms发送一次延迟探测包的工作线程
        
        Args:
            sock: MPTCP socket对象
        """
        print("[Sender] Delay probe worker started")
        
        while self.transfer_event.is_set():
            try:
                # 构造探测包: "DELAY_PROBE:" + 时间戳(8字节) + 序号(4字节) + ":END_PROBE"
                sent_timestamp = time.time()
                timestamp_us = int(sent_timestamp * 1000000)  # 转换为微秒精度
                
                # 打包二进制数据: ! = 网络字节序, Q = 8字节无符号整数, I = 4字节无符号整数
                probe_data = struct.pack('!QI', timestamp_us, self.probe_sequence)
                probe_packet = b'DELAY_PROBE:' + probe_data + b':END_PROBE'
                
                # 记录发送的探测包
                with self.delay_lock:
                    self.pending_probes[self.probe_sequence] = sent_timestamp
                    # 避免内存泄漏：删除超过10秒未回复的探测包
                    timeout_threshold = sent_timestamp - 10.0
                    timeout_seqs = [seq for seq, ts in self.pending_probes.items() if ts < timeout_threshold]
                    for seq in timeout_seqs:
                        del self.pending_probes[seq]
                
                # 发送探测包
                sock.send(probe_packet)
                self.probe_sequence += 1
                
                # 等待50ms后发送下一个
                time.sleep(self.probe_interval)
                
            except Exception as e:
                print(f"[Sender] Probe send error: {e}")
                break
                
        print("[Sender] Delay probe worker stopped")
    
    # 【新增】延迟回复接收工作线程  
    def delay_reply_worker(self, sock):
        """接收并处理延迟回复包的工作线程
        
        Args:
            sock: MPTCP socket对象
        """
        print("[Sender] Delay reply worker started")
        
        # 设置socket为非阻塞模式，避免影响文件传输
        sock.settimeout(0.1)  # 100ms超时
        buffer = b""
        
        while self.transfer_event.is_set():
            try:
                # 尝试接收数据
                data = sock.recv(1024)
                if not data:
                    continue
                    
                buffer += data
                
                # 在buffer中查找完整的回复包
                while b'DELAY_REPLY:' in buffer and b':END_REPLY' in buffer:
                    start = buffer.find(b'DELAY_REPLY:')
                    end = buffer.find(b':END_REPLY', start)
                    
                    if end != -1:
                        # 提取完整的回复包
                        reply_packet = buffer[start:end+10]  # +10 是":END_REPLY"的长度
                        buffer = buffer[end+10:]  # 剩余数据继续处理
                        
                        # 处理这个回复包
                        self.handle_delay_reply(reply_packet)
                    else:
                        break  # 数据不完整，等待更多数据
                        
            except socket.timeout:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                # 其他错误可能是连接断开
                print(f"[Sender] Reply receive error: {e}")
                break
                
        print("[Sender] Delay reply worker stopped")
    
    # 【新增】处理收到的延迟回复包
    def handle_delay_reply(self, reply_packet):
        """解析延迟回复包并计算延迟
        
        Args:
            reply_packet: 格式为 b'DELAY_REPLY:' + 数据 + b':END_REPLY'
        """
        try:
            # 提取数据部分 (去掉前缀"DELAY_REPLY:"和后缀":END_REPLY")
            payload = reply_packet[12:-10]  # 12="DELAY_REPLY:"长度, 10=":END_REPLY"长度
            
            # 解包数据: 时间戳(8字节) + 序号(4字节)
            sent_timestamp_us, sequence = struct.unpack('!QI', payload)
            sent_timestamp = sent_timestamp_us / 1000000.0  # 转换回秒
            received_timestamp = time.time()
            
            # 计算延迟(毫秒)
            delay_ms = (received_timestamp - sent_timestamp) * 1000
            
            with self.delay_lock:
                # 验证这个回复对应的探测包确实发送过
                if sequence in self.pending_probes:
                    del self.pending_probes[sequence]  # 移除pending记录
                    
                    # 存储延迟测量结果
                    self.delay_measurements.append((delay_ms, received_timestamp))
                    
                    # 保持最近100次测量，避免内存无限增长
                    if len(self.delay_measurements) > 100:
                        self.delay_measurements.pop(0)
                        
                    # 每10次测量打印一次统计
                    if len(self.delay_measurements) % 10 == 0:
                        recent = self.get_recent_delays(window_ms=150)
                        if recent:
                            avg_delay = sum(recent) / len(recent)
                            print(f"[Sender] Delay stats: {len(recent)} samples in 150ms, avg={avg_delay:.1f}ms")
                            
        except Exception as e:
            print(f"[Sender] Reply parse error: {e}")
        
    def run(self):
        """建立一次MPTCP连接，发送多个文件"""
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
            
            # 【新增】启动延迟探测相关线程
            probe_thread = threading.Thread(target=self.delay_probe_worker, args=(sock,))
            probe_thread.daemon = True  # 守护线程，主线程结束时自动结束
            probe_thread.start()
            
            reply_thread = threading.Thread(target=self.delay_reply_worker, args=(sock,))
            reply_thread.daemon = True
            reply_thread.start()
            
            # 【修改】启动Online Agent，传递sender引用
            agent = Online_Agent(fd=fd, cfg=self.cfg, memory=self.memory, 
                               event=self.transfer_event, sender=self)  # 传递self引用
            agent.start()
            self.transfer_event.set()
            
            # 依次发送每个文件
            for i, file_to_send in enumerate(self.file_list):
                start_time = time.time()
                
                try:
                    # 发送文件名
                    filename_msg = f"FILE:{file_to_send}\n".encode('utf-8')
                    sock.send(filename_msg)
                    
                    # 发送文件内容
                    with open(file_to_send, 'rb') as f:
                        while True:
                            data = f.read(4096)
                            if not data:
                                break
                            sock.sendall(data)
                    
                    end_time = time.time()
                    completion_time = end_time - start_time
                    
                    self.results.append({
                        'file': file_to_send,
                        'completion_time': completion_time,
                        'success': True
                    })
                    
                    print(f"[Sender] File {i+1}/{len(self.file_list)} sent: {file_to_send} ({completion_time:.2f}s)")
                    
                    # 文件间短暂间隔
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"[Sender] Error sending {file_to_send}: {e}")
                    self.results.append({
                        'file': file_to_send,
                        'completion_time': 0,
                        'success': False
                    })
            
        except socket.timeout:
            print("[Sender] Connection timeout")
            for file_to_send in self.file_list:
                self.results.append({
                    'file': file_to_send,
                    'completion_time': 0,
                    'success': False
                })
        except Exception as e:
            print(f"[Sender] Connection error: {e}")
            for file_to_send in self.file_list:
                self.results.append({
                    'file': file_to_send,
                    'completion_time': 0,
                    'success': False
                })
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
    FILE = cfg.get('file','file')
    FILES = ["64kb.dat","2mb.dat","8mb.dat","64mb.dat"]
    
    # 创建训练专用的event（永不清除）
    training_event = Event()
    training_event.set()
    CONTINUE_TRAIN = 1
    num_iterations = 150
    scenario = "default"
    batch_size = 10  # 每批发送的文件数
    
    # 解析命令行参数
    if len(argv) >= 1:
        CONTINUE_TRAIN = int(argv[0])
    if len(argv) >= 2:
        scenario = argv[1]
    if len(argv) >= 3:
        FILE = argv[2]
    if len(argv) >= 4:
        num_iterations = int(argv[3])
    if len(argv) >= 5:
        batch_size = int(argv[4])  # 新增：批大小参数
        
    now = datetime.now().replace(microsecond=0)
    start_train = now.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"[Sender] Starting with args CONTINUE_TRAIN={CONTINUE_TRAIN}, scenario='{scenario}'")
    print(f"[Sender] FILE={FILE}, iterations={num_iterations}, batch_size={batch_size}")
    
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
        # 主循环：分批发送文件
        total_files_sent = 0
        batch_count = 0
        
        while total_files_sent < num_iterations:
            batch_count += 1
            
            # 准备这一批要发送的文件
            current_batch_size = min(batch_size, num_iterations - total_files_sent)
            file_batch = []
            
            for i in range(current_batch_size):
                # 选择要发送的文件
                if FILE == "random" and num_iterations > 150:
                    FILE2 = np.random.choice(FILES,p=[0, 0.9, 0, 0.1])
                elif FILE == "random" and num_iterations == 150:
                    FILE2 = np.random.choice(FILES,p=[0.3,0.35,0.3,0.05])
                else:
                    FILE2 = FILE
                file_batch.append(FILE2)
            
            print(f"\n[Sender] Batch {batch_count}: Sending {len(file_batch)} files in one connection")
            
            # 记录开始时间
            batch_start_time = time.time()
            
            # 创建发送线程（一次连接发送多个文件）
            sender = MPTCPSender(cfg, memory, file_batch)
            sender.start()
            sender.join(timeout=60)  # 60秒超时
            
            if sender.is_alive():
                print(f"[Sender] Batch {batch_count} timeout, skipping")
                total_files_sent += len(file_batch)
                continue
            
            # 记录结束时间
            batch_end_time = time.time()
            
            # 处理结果
            for j, result in enumerate(sender.results):
                iteration_num = total_files_sent + j + 1
                
                if result['success']:
                    completion_time = result['completion_time']
                    file_name = result['file']
                    
                    # 计算性能指标
                    if file_name.find("kb") != -1:
                        file_size = int(re.findall(r'\d+',file_name)[0])/1000
                    else: 
                        file_size = int(re.findall(r'\d+',file_name)[0])
                        
                    throughput = file_size/completion_time if completion_time > 0 else 0
                    
                    # 保存性能指标
                    if iteration_num >= 30:
                        performance_metrics.append({
                            "iteration": iteration_num,
                            "file": file_name,
                            "completion time": completion_time,
                            "throughput": throughput,
                            "file_size_MB": file_size
                        })
                    
                    print(f"[Sender] File {iteration_num}: {file_name} - {throughput:.2f} MB/s")
                else:
                    print(f"[Sender] File {iteration_num}: {result['file']} - FAILED")
            
            total_files_sent += len(file_batch)
            
            # 检查是否需要启动离线训练
            if len(memory) > BATCH_SIZE:
                if off_agent is None or not off_agent.is_alive():
                    off_agent = Offline_Agent(cfg=cfg, model=AGENT_FILE, 
                                             memory=memory, event=training_event)
                    off_agent.daemon = True
                    off_agent.start()
                    print(f"[Sender] Training started/restarted after batch {batch_count}")
                    print(f"[Sender] Memory size: {len(memory)}")
            
            # 批次间间隔（给系统时间清理资源）
            time.sleep(2)
            
        # 保存性能指标
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
        # 保存replay memory
        with open(MEMORY_FILE,'wb') as f:
            pickle.dump(memory,f)
            f.close()
        print("[Sender] Memory saved")

if __name__ == '__main__':
    main(sys.argv[1:])