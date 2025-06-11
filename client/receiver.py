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
        """接收并保存文件"""
        try:
            # 首先接收文件名
            header = b""
            while b"\n" not in header:
                data = self.conn.recv(1)
                if not data:
                    return
                header += data
                
            header_str = header.decode('utf-8').strip()
            if header_str.startswith("FILE:"):
                filename = header_str[5:]
                print(f"[Receiver] Receiving '{filename}' from {self.addr}")
                
                # 确保保存目录存在
                os.makedirs(self.save_dir, exist_ok=True)
                filepath = os.path.join(self.save_dir, filename)
                
                # 接收文件内容
                start_time = time.time()
                total_bytes = 0
                
                with open(filepath, 'wb') as f:
                    while True:
                        data = self.conn.recv(4096)
                        if not data:
                            break
                        f.write(data)
                        total_bytes += len(data)
                
                end_time = time.time()
                duration = end_time - start_time
                
                # 计算接收速率
                if duration > 0:
                    rate_mbps = (total_bytes * 8) / (duration * 1000000)
                    print(f"[Receiver] Completed '{filename}': {total_bytes} bytes in {duration:.2f}s ({rate_mbps:.2f} Mbps)")
                else:
                    print(f"[Receiver] Completed '{filename}': {total_bytes} bytes")
                    
        except Exception as e:
            print(f"[Receiver] Error handling connection from {self.addr}: {e}")
        finally:
            self.conn.close()

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
