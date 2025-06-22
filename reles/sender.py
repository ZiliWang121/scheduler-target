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
import select  # 【新增】用于非阻塞socket操作

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
        
        # 【修复】延迟测量相关属性 - 重新设计数据结构
        self.delay_measurements = []  # 存储[(delay_ms, received_timestamp), ...]
        self.delay_lock = threading.Lock()  # 线程安全锁
        self.probe_sequence = 0  # 探测包序号
        self.probe_interval = 0.05  # 50ms发送一次探测包
        self.delay_active = False  # 控制延迟测量线程
        
        # 【新增】用于socket数据分离的缓冲区
        self.socket_buffer = b""
        self.socket_lock = threading.Lock()
        
    def get_recent_delays(self, window_ms=150):
        """
        【强制One-Way Delay】获取最近window_ms毫秒内收到的延迟测量，供Env调用
        
        Args:
            window_ms: 时间窗口，默认150ms
            
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
            
            # 【强制验证】：检查延迟测量系统是否正常工作
            total_measurements = len(self.delay_measurements)
            if total_measurements == 0:
                print(f"[Sender] *** CRITICAL: NO DELAY MEASUREMENTS RECEIVED AT ALL ***")
                print(f"[Sender] *** PING system appears to be completely broken ***")
            elif len(recent_delays) == 0 and total_measurements > 0:
                latest_time = max(received_time for _, received_time in self.delay_measurements) if self.delay_measurements else 0
                age = current_time - latest_time
                print(f"[Sender] *** WARNING: No recent delays in {window_ms}ms window ***")
                print(f"[Sender] *** Total measurements: {total_measurements}, latest age: {age:.3f}s ***")
            
        return recent_delays
    
    def delay_probe_worker(self, sock):
        """
        【修复】每50ms在MPTCP连接中发送一次延迟探测包的工作线程
        修复要点：
        1. 确保探测包格式正确
        2. 控制发送频率
        3. 添加详细的调试信息
        """
        print("[Sender] Delay probe worker started")
        
        # 等待传输开始后再开始探测
        while not self.transfer_event.is_set():
            time.sleep(0.1)
        
        # 【修复】延迟启动，让连接和文件传输先稳定
        time.sleep(2.0)  # 增加到2秒
        
        self.delay_active = True
        probe_count = 0
        max_probes = 1000  # 最多发送1000个探测包
        
        print("[Sender] Starting delay probe transmission")
        
        while self.transfer_event.is_set() and self.delay_active and probe_count < max_probes:
            try:
                # 【修复】构造探测包格式：PING:timestamp:sequence\n
                sent_timestamp = time.time()
                probe_msg = f"PING:{sent_timestamp:.6f}:{self.probe_sequence}\n"
                probe_data = probe_msg.encode('utf-8')
                
                # 【关键修复】使用send而不是sendall，并添加错误处理
                try:
                    bytes_sent = sock.send(probe_data)
                    if bytes_sent == len(probe_data):
                        self.probe_sequence += 1
                        probe_count += 1
                        
                        # 【调试】前10个包打印详细信息
                        if probe_count <= 10:
                            print(f"[Sender] Sent probe #{self.probe_sequence-1}: {sent_timestamp:.6f}")
                        elif probe_count % 50 == 0:
                            print(f"[Sender] Sent {probe_count} delay probes")
                    else:
                        print(f"[Sender] Warning: Probe send incomplete {bytes_sent}/{len(probe_data)}")
                        
                except socket.error as e:
                    print(f"[Sender] Probe send socket error: {e}")
                    time.sleep(0.1)  # 短暂等待后继续
                    continue
                
                # 【修复】精确控制发送间隔
                time.sleep(self.probe_interval)
                
            except Exception as e:
                print(f"[Sender] Probe worker unexpected error: {e}")
                break
                
        print(f"[Sender] Delay probe worker stopped after {probe_count} probes")
        self.delay_active = False
    
    def unified_socket_reader(self, sock):
        """
        【新增】统一的socket数据读取器，负责分离文件数据和延迟回复
        这是修复的关键：将所有socket接收逻辑集中到一个地方
        """
        print("[Sender] Unified socket reader started")
        
        buffer = b""
        delay_reply_count = 0
        
        while self.transfer_event.is_set():
            try:
                # 【修复】使用select进行非阻塞读取，超时100ms
                ready, _, _ = select.select([sock], [], [], 0.1)
                
                if not ready:
                    continue  # 没有数据可读，继续循环
                
                # 有数据可读，接收数据
                try:
                    data = sock.recv(4096)
                    if not data:
                        print("[Sender] Socket closed by receiver")
                        break
                        
                    buffer += data
                    
                    # 【关键修复】处理缓冲区中的延迟回复
                    while b'\n' in buffer:
                        line_end = buffer.find(b'\n')
                        line = buffer[:line_end].decode('utf-8', errors='ignore')
                        buffer = buffer[line_end + 1:]
                        
                        # 【修复】检查是否是DELAY回复
                        if line.startswith('DELAY:'):
                            self.handle_delay_reply(line)
                            delay_reply_count += 1
                            
                            # 【调试】前10个回复打印详细信息
                            if delay_reply_count <= 10:
                                print(f"[Sender] Processed DELAY reply #{delay_reply_count}: {line[:50]}...")
                            elif delay_reply_count % 20 == 0:
                                recent = self.get_recent_delays(window_ms=150)
                                if recent:
                                    avg_delay = sum(recent) / len(recent)
                                    print(f"[Sender] Processed {delay_reply_count} replies, recent avg delay: {avg_delay:.1f}ms")
                        # 【注意】非DELAY回复的数据（如果有）会被忽略，这在当前设计中是正确的
                        # 因为文件传输是单向的，receiver不应该发送其他类型的回复
                        
                except socket.error as e:
                    print(f"[Sender] Socket read error: {e}")
                    time.sleep(0.1)
                    continue
                        
            except Exception as e:
                print(f"[Sender] Unified reader unexpected error: {e}")
                break
                
        print(f"[Sender] Unified socket reader stopped, processed {delay_reply_count} delay replies")
    
    def handle_delay_reply(self, reply_line):
        """
        【强制One-Way Delay】处理receiver发回的one-way delay测量结果
        修复要点：
        1. 更严格的格式检查
        2. 更好的错误处理
        3. 延迟合理性验证
        4. 强制性的成功确认
        
        Args:
            reply_line: 格式为 "DELAY:delay_ms:sequence"
        """
        try:
            parts = reply_line.split(':')
            if len(parts) >= 3:
                delay_ms = float(parts[1])  # receiver计算好的延迟值(ms)
                sequence = int(parts[2])
                
                received_timestamp = time.time()
                
                # 【强制验证】：更严格的延迟合理性检查
                if 0.1 < delay_ms < 5000:  # 延迟应该在0.1ms-5秒之间
                    with self.delay_lock:
                        # 存储延迟测量结果
                        self.delay_measurements.append((delay_ms, received_timestamp))
                        
                        # 【优化】保持最近200次测量（增加容量）
                        if len(self.delay_measurements) > 200:
                            self.delay_measurements.pop(0)
                        
                        # 【强制确认】：前10个有效回复打印详细信息
                        if len(self.delay_measurements) <= 10:
                            print(f"[Sender] ✓ VALID ONE-WAY DELAY #{sequence}: {delay_ms:.3f}ms (total: {len(self.delay_measurements)})")
                        elif len(self.delay_measurements) % 20 == 0:
                            recent = self.get_recent_delays(window_ms=500)  # 检查最近500ms
                            if recent:
                                avg_delay = sum(recent) / len(recent)
                                print(f"[Sender] ✓ PING SYSTEM WORKING: {len(self.delay_measurements)} total, recent avg: {avg_delay:.1f}ms")
                else:
                    print(f"[Sender] *** INVALID DELAY: {delay_ms:.1f}ms (seq #{sequence}) - outside valid range ***")
            else:
                print(f"[Sender] *** MALFORMED DELAY REPLY: '{reply_line}' - wrong format ***")
                            
        except (ValueError, IndexError) as e:
            print(f"[Sender] *** DELAY PARSING ERROR: '{reply_line}' - {e} ***")
        
    def run(self):
        """【修复】建立一次MPTCP连接，发送多个文件，同时进行延迟测量"""
        # 创建MPTCP socket
        MPTCP_ENABLED = 42
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 添加连接超时
        
        # 启用MPTCP
        try:
            sock.setsockopt(socket.IPPROTO_TCP, MPTCP_ENABLED, 1)
            print("[Sender] MPTCP enabled successfully")
        except Exception as e:
            print(f"[Sender] Warning: Could not enable MPTCP: {e}")
        
        try:
            # 建立连接
            print(f"[Sender] Connecting to {self.IP}:{self.PORT}")
            sock.connect((self.IP, self.PORT))
            sock.settimeout(None)  # 连接后取消超时
            fd = sock.fileno()
            mpsched.persist_state(fd)
            print("[Sender] MPTCP connection established")
            
            # 【修改】启动Online Agent，传递sender引用
            agent = Online_Agent(fd=fd, cfg=self.cfg, memory=self.memory, 
                               event=self.transfer_event, sender=self)  # 传递self引用
            agent.start()
            self.transfer_event.set()
            
            # 【关键修复】启动统一的socket读取器（替代原来的reply worker）
            reader_thread = threading.Thread(target=self.unified_socket_reader, args=(sock,))
            reader_thread.daemon = True
            reader_thread.start()
            
            # 【修复】启动延迟探测线程
            probe_thread = threading.Thread(target=self.delay_probe_worker, args=(sock,))
            probe_thread.daemon = True
            probe_thread.start()
            
            # 依次发送每个文件
            for i, file_to_send in enumerate(self.file_list):
                start_time = time.time()
                
                try:
                    # 发送文件名
                    filename_msg = f"FILE:{file_to_send}\n".encode('utf-8')
                    sock.send(filename_msg)
                    print(f"[Sender] Sending file header: {file_to_send}")
                    
                    # 发送文件内容 - 与延迟探测包共享同一连接
                    with open(file_to_send, 'rb') as f:
                        bytes_sent = 0
                        while True:
                            data = f.read(4096)
                            if not data:
                                break
                            sock.sendall(data)
                            bytes_sent += len(data)
                    
                    end_time = time.time()
                    completion_time = end_time - start_time
                    
                    self.results.append({
                        'file': file_to_send,
                        'completion_time': completion_time,
                        'success': True
                    })
                    
                    print(f"[Sender] File {i+1}/{len(self.file_list)} sent: {file_to_send} "
                          f"({bytes_sent} bytes in {completion_time:.2f}s)")
                    
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
            # 【修复】正确清理资源
            print("[Sender] Cleaning up connection")
            self.delay_active = False
            self.transfer_event.clear()
            time.sleep(0.5)  # 等待线程结束
            sock.close()
            
            # 【调试】打印最终的延迟测量统计
            with self.delay_lock:
                total_measurements = len(self.delay_measurements)
                if total_measurements > 0:
                    recent_delays = self.get_recent_delays(window_ms=1000)  # 最近1秒的数据
                    if recent_delays:
                        avg_delay = sum(recent_delays) / len(recent_delays)
                        print(f"[Sender] Final delay stats: {total_measurements} total measurements, "
                              f"recent avg: {avg_delay:.1f}ms")
                    else:
                        print(f"[Sender] Final delay stats: {total_measurements} total measurements, no recent data")
                else:
                    print("[Sender] Final delay stats: No delay measurements received")

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
        batch_size = int(argv[4])
        
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