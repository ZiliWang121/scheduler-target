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
                
                # 修改1: 为避免文件覆盖，添加时间戳到文件名
                timestamp = int(time.time() * 1000) % 100000
                base_name, ext = os.path.splitext(filename)
                safe_filename = f"{base_name}_{timestamp}{ext}"
                filepath = os.path.join(self.save_dir, safe_filename)
                
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
                    rate_MB_s = total_bytes / (duration * 1024 * 1024)  # 修改2: 添加MB/s计算
                    print(f"[Receiver] Completed '{filename}': {total_bytes} bytes in {duration:.2f}s ({rate_MB_s:.2f} MB/s, {rate_mbps:.2f} Mbps)")
                else:
                    print(f"[Receiver] Completed '{filename}': {total_bytes} bytes")
                    
        except Exception as e:
            print(f"[Receiver] Error handling connection from {self.addr}: {e}")
        finally:
            self.conn.close()
            # 修改3: 添加连接关闭日志，便于调试
            print(f"[Receiver] Connection from {self.addr} closed")

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
    # 修改4: 添加说明，每个连接接收一个文件
    print("[Receiver] Each connection will receive exactly one file")
    
    # 修改5: 添加连接计数器，便于跟踪
    connection_count = 0
    
    try:
        while True:
            # 接受连接
            conn, addr = server_sock.accept()
            connection_count += 1
            print(f"[Receiver] Connection #{connection_count} from {addr}")
            
            # 为每个连接创建新线程
            handler = ConnectionHandler(conn, addr)
            handler.daemon = True
            handler.start()
            
    except KeyboardInterrupt:
        print(f"\n[Receiver] Shutting down after {connection_count} connections...")
    finally:
        server_sock.close()

if __name__ == '__main__':
    main()