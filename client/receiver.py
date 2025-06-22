#!/usr/bin/python3

import socket
import sys
import os
import threading
from configparser import ConfigParser
import time
import struct  # 【新增】用于处理二进制数据

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
            
            print(f"[Receiver] Starting to handle connection from {self.addr}")
            
            while True:
                data = self.conn.recv(4096)
                if not data:
                    print(f"[Receiver] Connection closed by {self.addr}")
                    break
                    
                buffer += data
                
                # 【新增】处理buffer中混合的数据：文件数据 + 延迟探测包
                while buffer:
                    if not current_file:
                        # 当前没有在接收文件，查找文件头或延迟探测包
                        if b'DELAY_PROBE:' in buffer:
                            # 处理延迟探测包
                            buffer = self.handle_delay_probe(buffer)
                        elif b'FILE:' in buffer and b'\n' in buffer:
                            # 处理文件头
                            newline_pos = buffer.find(b'\n')
                            header = buffer[:newline_pos].decode('utf-8')
                            buffer = buffer[newline_pos+1:]
                            
                            if header.startswith('FILE:'):
                                filename = header[5:]
                                print(f"[Receiver] Starting to receive file '{filename}'")
                                current_file = filename
                                
                                # 确保保存目录存在
                                os.makedirs(self.save_dir, exist_ok=True)
                                filepath = os.path.join(self.save_dir, filename)
                                current_file_obj = open(filepath, 'wb')
                        else:
                            # 数据不完整，等待更多数据
                            break
                    else:
                        # 当前正在接收文件，检查数据中是否混有延迟探测包
                        probe_start = buffer.find(b'DELAY_PROBE:')
                        if probe_start != -1:
                            # 先写入探测包之前的文件数据
                            if probe_start > 0:
                                current_file_obj.write(buffer[:probe_start])
                                buffer = buffer[probe_start:]
                            
                            # 处理延迟探测包
                            buffer = self.handle_delay_probe(buffer)
                        else:
                            # 检查是否有新的文件头（表示当前文件结束）
                            file_start = buffer.find(b'FILE:')
                            if file_start != -1:
                                # 写入文件头之前的数据，结束当前文件
                                current_file_obj.write(buffer[:file_start])
                                buffer = buffer[file_start:]
                                
                                # 关闭当前文件
                                current_file_obj.close()
                                print(f"[Receiver] Completed receiving file '{current_file}'")
                                current_file = None
                                current_file_obj = None
                            else:
                                # 全部是文件数据
                                current_file_obj.write(buffer)
                                buffer = b""
                                
        except Exception as e:
            print(f"[Receiver] Error handling connection from {self.addr}: {e}")
        finally:
            if current_file_obj:
                current_file_obj.close()
                print(f"[Receiver] File '{current_file}' closed due to connection end")
            self.conn.close()
    
    def handle_delay_probe(self, buffer):
        """
        【新增】处理延迟探测包并立即发送回复
        
        Args:
            buffer: 包含延迟探测包的数据缓冲区
            
        Returns:
            bytes: 处理后剩余的缓冲区数据
        """
        try:
            start = buffer.find(b'DELAY_PROBE:')
            end = buffer.find(b':END_PROBE', start)
            
            if end != -1:
                # 提取完整的探测包
                probe_packet = buffer[start:end+10]  # +10 是":END_PROBE"的长度
                remaining_buffer = buffer[end+10:]   # 剩余数据
                
                # 提取探测包中的数据部分 (去掉前缀"DELAY_PROBE:"和后缀":END_PROBE")
                payload = probe_packet[12:-10]  # 12="DELAY_PROBE:"长度, 10=":END_PROBE"长度
                
                # 构造回复包：使用相同的payload，只是把前缀改为"DELAY_REPLY:"
                reply_packet = b'DELAY_REPLY:' + payload + b':END_REPLY'
                
                # 立即发送回复包
                self.conn.send(reply_packet)
                
                # 可选：统计延迟探测包的处理情况
                if hasattr(self, 'probe_count'):
                    self.probe_count += 1
                else:
                    self.probe_count = 1
                    
                # 每处理100个探测包打印一次统计
                if self.probe_count % 100 == 0:
                    print(f"[Receiver] Processed {self.probe_count} delay probes from {self.addr}")
                
                return remaining_buffer
            else:
                # 探测包数据不完整，保留整个buffer等待更多数据
                return buffer
                
        except Exception as e:
            print(f"[Receiver] Error handling delay probe: {e}")
            # 出错时跳过可能有问题的探测包
            if b':END_PROBE' in buffer:
                end_pos = buffer.find(b':END_PROBE') + 10
                return buffer[end_pos:]
            else:
                return buffer

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
    print("[Receiver] Ready to receive files and handle delay probes...")
    
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