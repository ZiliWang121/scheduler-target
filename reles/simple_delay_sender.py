#!/usr/bin/python3
#python3 simple_delay_sender.py 10.0.1.10 8888 30 1  #mptcp
#python3 simple_delay_sender.py 10.0.1.10 8888 30 0  #tcp
import socket
import time
import sys
import threading
import select
from statistics import mean, median, stdev

class DelayMeasurementSender:
    def __init__(self, target_ip, target_port, use_mptcp=True):
        self.target_ip = target_ip
        self.target_port = target_port
        self.use_mptcp = use_mptcp
        
        # 延迟测量相关
        self.delay_measurements = []  # [(delay_ms, sequence), ...]
        self.probe_sequence = 0
        self.probe_interval = 1  # 50ms发送间隔
        self.measurement_active = False
        
        # 统计相关
        self.sent_count = 0
        self.received_count = 0
        self.lost_count = 0
        
    def create_socket(self):
        """创建socket连接"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        if self.use_mptcp:
            try:
                MPTCP_ENABLED = 42
                sock.setsockopt(socket.IPPROTO_TCP, MPTCP_ENABLED, 1)
                print(f"[Sender] MPTCP enabled")
            except Exception as e:
                print(f"[Sender] Warning: Could not enable MPTCP: {e}")
                print(f"[Sender] Using regular TCP")
        else:
            print(f"[Sender] Using regular TCP")
            
        return sock
    
    def send_probe_worker(self, sock):
        """发送延迟探测包的工作线程"""
        print(f"[Sender] Starting delay probe transmission")
        self.measurement_active = True
        
        while self.measurement_active:
            try:
                # 记录发送时间并构造探测包
                sent_timestamp = time.time()
                probe_msg = 
f"PING:{sent_timestamp:.6f}:{self.probe_sequence}\n"
                probe_data = probe_msg.encode('utf-8')
                
                # 发送探测包
                try:
                    bytes_sent = sock.send(probe_data)
                    if bytes_sent == len(probe_data):
                        self.sent_count += 1
                        self.probe_sequence += 1
                        
                        # 前10个包打印详细信息
                        if self.sent_count <= 10:
                            print(f"[Sender] Sent probe 
#{self.probe_sequence-1}: {sent_timestamp:.6f}")
                        elif self.sent_count % 50 == 0:
                            print(f"[Sender] Sent {self.sent_count} 
probes")
                            
                except socket.error as e:
                    print(f"[Sender] Send error: {e}")
                    break
                
                # 控制发送间隔
                time.sleep(self.probe_interval)
                
            except Exception as e:
                print(f"[Sender] Probe worker error: {e}")
                break
                
        print(f"[Sender] Probe worker stopped after sending 
{self.sent_count} probes")
    
    def receive_reply_worker(self, sock):
        """接收延迟回复的工作线程"""
        print(f"[Sender] Starting reply receiver")
        buffer = b""
        
        while self.measurement_active:
            try:
                # 非阻塞读取
                ready, _, _ = select.select([sock], [], [], 0.1)
                
                if not ready:
                    continue
                
                try:
                    data = sock.recv(4096)
                    if not data:
                        print(f"[Sender] Connection closed by receiver")
                        break
                        
                    buffer += data
                    
                    # 处理缓冲区中的DELAY回复
                    while b'\n' in buffer:
                        line_end = buffer.find(b'\n')
                        line = buffer[:line_end].decode('utf-8', 
errors='ignore')
                        buffer = buffer[line_end + 1:]
                        
                        if line.startswith('DELAY:'):
                            self.handle_delay_reply(line)
                            
                except socket.error as e:
                    print(f"[Sender] Receive error: {e}")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[Sender] Reply worker error: {e}")
                break
                
        print(f"[Sender] Reply worker stopped, received 
{self.received_count} replies")
    
    def handle_delay_reply(self, reply_line):
        """处理延迟回复"""
        try:
            parts = reply_line.split(':')
            if len(parts) >= 3:
                delay_ms = float(parts[1])
                sequence = int(parts[2])
                
                # 验证延迟合理性
                if 0.01 < delay_ms < 10000:  # 0.01ms - 10秒
                    self.delay_measurements.append((delay_ms, sequence))
                    self.received_count += 1
                    
                    # 前10个回复打印详细信息
                    if self.received_count <= 10:
                        print(f"[Sender] Received DELAY #{sequence}: 
{delay_ms:.3f}ms")
                    elif self.received_count % 20 == 0:
                        recent_delays = [d[0] for d in 
self.delay_measurements[-20:]]
                        avg_delay = mean(recent_delays)
                        print(f"[Sender] Received {self.received_count} 
replies, recent avg: {avg_delay:.3f}ms")
                else:
                    print(f"[Sender] Invalid delay: {delay_ms:.1f}ms (seq 
#{sequence})")
            else:
                print(f"[Sender] Malformed reply: '{reply_line}'")
                
        except (ValueError, IndexError) as e:
            print(f"[Sender] Reply parsing error: '{reply_line}' - {e}")
    
    def run_measurement(self, duration_seconds=30):
        """运行延迟测量"""
        print(f"[Sender] Starting {duration_seconds}s delay measurement to 
{self.target_ip}:{self.target_port}")
        
        # 创建连接
        sock = self.create_socket()
        try:
            sock.connect((self.target_ip, self.target_port))
            print(f"[Sender] Connected successfully")
            
            # 启动工作线程
            send_thread = threading.Thread(target=self.send_probe_worker, 
args=(sock,))
            recv_thread = 
threading.Thread(target=self.receive_reply_worker, args=(sock,))
            
            send_thread.daemon = True
            recv_thread.daemon = True
            
            send_thread.start()
            recv_thread.start()
            
            # 等待指定时间
            time.sleep(duration_seconds)
            
        except Exception as e:
            print(f"[Sender] Connection error: {e}")
        finally:
            # 停止测量
            self.measurement_active = False
            time.sleep(0.5)  # 等待线程结束
            sock.close()
            
            # 计算统计结果
            self.calculate_statistics()
    
    def calculate_statistics(self):
        """计算并打印统计结果"""
        print(f"\n{'='*60}")
        print(f"DELAY MEASUREMENT RESULTS")
        print(f"{'='*60}")
        
        print(f"Probes sent: {self.sent_count}")
        print(f"Replies received: {self.received_count}")
        
        if self.sent_count > 0:
            loss_rate = ((self.sent_count - self.received_count) / 
self.sent_count) * 100
            print(f"Loss rate: {loss_rate:.1f}%")
        
        if self.delay_measurements:
            delays = [d[0] for d in self.delay_measurements]
            
            print(f"\nDelay Statistics (ms):")
            print(f"  Samples: {len(delays)}")
            print(f"  Min: {min(delays):.3f}")
            print(f"  Max: {max(delays):.3f}")
            print(f"  Mean: {mean(delays):.3f}")
            print(f"  Median: {median(delays):.3f}")
            
            if len(delays) > 1:
                print(f"  Std Dev: {stdev(delays):.3f}")
            
            # 分位数
            sorted_delays = sorted(delays)
            n = len(sorted_delays)
            print(f"  95th percentile: {sorted_delays[int(n * 
0.95)]:.3f}")
            print(f"  99th percentile: {sorted_delays[int(n * 
0.99)]:.3f}")
            
            # 分布统计
            ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 
100), (100, 200), (200, float('inf'))]
            print(f"\nDelay Distribution:")
            for min_val, max_val in ranges:
                count = sum(1 for d in delays if min_val <= d < max_val)
                if count > 0:
                    percentage = (count / len(delays)) * 100
                    if max_val == float('inf'):
                        print(f"  {min_val}ms+: {count} 
({percentage:.1f}%)")
                    else:
                        print(f"  {min_val}-{max_val}ms: {count} 
({percentage:.1f}%)")
        else:
            print(f"\nNo valid delay measurements received!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 simple_delay_sender.py <target_ip> 
<target_port> [duration] [mptcp]")
        print("  target_ip: receiver IP address")
        print("  target_port: receiver port")
        print("  duration: measurement duration in seconds (default: 30)")
        print("  mptcp: use MPTCP (1) or TCP (0) (default: 1)")
        sys.exit(1)
    
    target_ip = sys.argv[1]
    target_port = int(sys.argv[2])
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    use_mptcp = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    
    sender = DelayMeasurementSender(target_ip, target_port, use_mptcp)
    sender.run_measurement(duration)

if __name__ == '__main__':
    main()

