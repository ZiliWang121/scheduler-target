#!/usr/bin/python3

import socket
import sys
import os
import threading
from configparser import ConfigParser
import time

class ConnectionHandler(threading.Thread):
    """处理单个连接的线程"""
    def __init__(self, conn, addr, save_dir="received_files"):
        threading.Thread.__init__(self)
        self.conn = conn
        self.addr = addr
        self.save_dir = save_dir
        
    def run(self):
        """接收并保存文件，同时处理延迟探测包"""
        try:
            buffer = b""
            current_file = None
            current_file_obj = None
            file_start_time = None
            total_bytes = 0
            
            print(f"[Receiver] Starting to handle connection from {self.addr}")
            
            while True:
                data = self.conn.recv(4096)
                if not data:
                    print(f"[Receiver] Connection closed by {self.addr}")
                    break
                    
                buffer += data
                
                # 【新增】优先处理延迟探测包
                buffer = self.process_delay_probes(buffer)
                
                # 【修改】处理文件数据
                result = self.process_file_data(buffer, current_file, current_file_obj, file_start_time, total_bytes)
                buffer, current_file, current_file_obj, file_start_time, total_bytes = result
                                
        except Exception as e:
            print(f"[Receiver] Error handling connection from {self.addr}: {e}")
        finally:
            if current_file_obj:
                if file_start_time:
                    file_end_time = time.time()
                    duration = file_end_time - file_start_time
                    if duration > 0:
                        rate_mbps = (total_bytes * 8) / (duration * 1000000)
                        print(f"[Receiver] File '{current_file}' closed: {total_bytes} bytes in {duration:.2f}s ({rate_mbps:.2f} Mbps)")
                    else:
                        print(f"[Receiver] File '{current_file}' closed: {total_bytes} bytes")
                current_file_obj.close()
            self.conn.close()
    
    def process_delay_probes(self, buffer):
        """
        处理缓冲区中的延迟探测包并立即发送回复
        
        Args:
            buffer: 包含延迟探测包的数据缓冲区
            
        Returns:
            bytes: 处理后剩余的缓冲区数据
        """
        probe_count = 0
        
        while True:
            # 查找PING行
            if b'PING:' not in buffer:
                break
                
            ping_start = buffer.find(b'PING:')
            newline_pos = buffer.find(b'\n', ping_start)
            
            if newline_pos == -1:
                break  # PING行不完整，等待更多数据
                
            # 提取完整的PING行
            ping_line = buffer[ping_start:newline_pos].decode('utf-8', errors='ignore')
            
            # 从缓冲区移除已处理的PING行
            buffer = buffer[:ping_start] + buffer[newline_pos + 1:]
            
            # 【修正】处理PING并计算one-way delay，然后回复结果
            try:
                # 解析PING: "PING:timestamp_seconds:sequence"
                parts = ping_line.split(':')
                if len(parts) >= 3:
                    timestamp_seconds = float(parts[1])  # 发送时间戳
                    sequence = parts[2]
                    
                    # 【关键】计算one-way delay
                    receive_time = time.time()
                    delay_ms = (receive_time - timestamp_seconds) * 1000  # 转换为毫秒
                    
                    # 【调试】前几个包打印详细时间戳信息
                    if probe_count < 5:
                        print(f"[Receiver] PING #{sequence}: send_time={timestamp_seconds:.6f}, "
                              f"receive_time={receive_time:.6f}, "
                              f"one_way_delay={delay_ms:.3f}ms")
                    
                    # 构造DELAY回复: "DELAY:delay_ms:sequence\n"
                    delay_msg = f"DELAY:{delay_ms:.3f}:{sequence}\n"
                    delay_data = delay_msg.encode('utf-8')
                    
                    # 立即发送回复
                    self.conn.send(delay_data)
                    probe_count += 1
                    
            except Exception as e:
                print(f"[Receiver] Error processing PING: {ping_line}, error: {e}")
        
        # 统计延迟探测包的处理情况
        if probe_count > 0:
            if not hasattr(self, 'total_probe_count'):
                self.total_probe_count = 0
            self.total_probe_count += probe_count
            
            # 每100个探测包打印一次统计
            if self.total_probe_count % 100 == 0:
                print(f"[Receiver] Processed {self.total_probe_count} delay probes from {self.addr}")
        
        return buffer
    
    def process_file_data(self, buffer, current_file, current_file_obj, file_start_time, total_bytes):
        """
        处理文件数据部分
        
        Returns:
            tuple: (buffer, current_file, current_file_obj, file_start_time, total_bytes)
        """
        while buffer:
            if not current_file:
                # 当前没有在接收文件，查找文件头
                if b'FILE:' in buffer and b'\n' in buffer:
                    # 处理文件头
                    file_start = buffer.find(b'FILE:')
                    newline_pos = buffer.find(b'\n', file_start)
                    header = buffer[file_start:newline_pos].decode('utf-8')
                    buffer = buffer[newline_pos+1:]
                    
                    if header.startswith('FILE:'):
                        filename = header[5:]
                        print(f"[Receiver] Starting to receive file '{filename}'")
                        current_file = filename
                        
                        # 确保保存目录存在
                        os.makedirs(self.save_dir, exist_ok=True)
                        filepath = os.path.join(self.save_dir, filename)
                        current_file_obj = open(filepath, 'wb')
                        
                        # 重置计时和计数
                        file_start_time = time.time()
                        total_bytes = 0
                else:
                    # 数据不完整，等待更多数据
                    break
            else:
                # 当前正在接收文件，检查是否有新的文件头（表示当前文件结束）
                file_start = buffer.find(b'FILE:')
                if file_start != -1:
                    # 写入文件头之前的数据，结束当前文件
                    current_file_obj.write(buffer[:file_start])
                    total_bytes += file_start
                    buffer = buffer[file_start:]
                    
                    # 计算并显示传输统计
                    file_end_time = time.time()
                    duration = file_end_time - file_start_time
                    if duration > 0:
                        rate_mbps = (total_bytes * 8) / (duration * 1000000)
                        print(f"[Receiver] Completed '{current_file}': {total_bytes} bytes in {duration:.2f}s ({rate_mbps:.2f} Mbps)")
                    else:
                        print(f"[Receiver] Completed '{current_file}': {total_bytes} bytes")
                    
                    # 关闭当前文件，重置状态
                    current_file_obj.close()
                    current_file = None
                    current_file_obj = None
                    file_start_time = None
                    total_bytes = 0
                else:
                    # 全部是文件数据
                    current_file_obj.write(buffer)
                    total_bytes += len(buffer)
                    buffer = b""
                    break
                    
        return buffer, current_file, current_file_obj, file_start_time, total_bytes

def main():
    cfg = ConfigParser()
    cfg.read('config.ini')
    
    # 读取配置
    LISTEN_IP = cfg.get('receiver','listen_ip')
    LISTEN_PORT = cfg.getint('receiver','listen_port')
    
    # 创建监听socket
    MPTCP_ENABLED = 42  # 根据您的内核版本可能需要调整
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 尝试启用MPTCP
    try:
        server_sock.setsockopt(socket.IPPROTO_TCP, MPTCP_ENABLED, 1)
        print("[Receiver] MPTCP enabled")
    except:
        print("[Receiver] Warning: Could not enable MPTCP, using regular TCP")
    
    # 设置socket选项
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 绑定和监听
    server_sock.bind((LISTEN_IP, LISTEN_PORT))
    server_sock.listen(10)
    
    print(f"[Receiver] Listening on {LISTEN_IP}:{LISTEN_PORT}")
    print("[Receiver] Waiting for connections...")
    
    try:
        while True:
            # 接受连接
            conn, addr = server_sock.accept()
            print(f"[Receiver] New connection from {addr}")
            
            # 为每个连接创建新线程
            handler = ConnectionHandler(conn, addr)
            handler.daemon = True
            handler.start()
            
    except KeyboardInterrupt:
        print("\n[Receiver] Shutting down...")
    finally:
        server_sock.close()

if __name__ == '__main__':
    main()
