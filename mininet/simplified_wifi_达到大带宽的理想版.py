#!/usr/bin/env python

from mn_wifi.cli import CLI
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mininet.node import Controller
import os
import time

def create_dual_wifi_network():
    """创建双WiFi网络，修复路由问题"""
    os.system('sudo mn -c')
    setLogLevel('info')
    
    # 不使用wmediumd，获得最大带宽
    net = Mininet_wifi(controller=Controller)

    # 网络节点
    sta1 = net.addStation('sta1', wlans=2, ip='10.0.1.1', position='50,50,0')
    h1 = net.addHost('h1', ip='10.0.1.10')
    ap1 = net.addAccessPoint('ap1', ssid='ssid1', mode='a', channel=36, position='50,60,0')
    ap2 = net.addAccessPoint('ap2', ssid='ssid2', mode='g', channel=1, position='50,40,0')
    c0 = net.addController('c0')
    s1 = net.addSwitch('s1')

    # 理想传播模型，无信号衰减
    net.setPropagationModel(model="constant")
    net.configureNodes()

    # 建立链路
    net.addLink(sta1, ap1, 0, 0)
    net.addLink(sta1, ap2, 1, 0)
    net.addLink(ap1, s1)
    net.addLink(ap2, s1)
    net.addLink(s1, h1)

    # 启动网络
    net.build()
    c0.start()
    ap1.start([c0])
    ap2.start([c0])
    s1.start([c0])
    time.sleep(10)
    
    # 配置IP和WiFi连接
    sta1.setIP('10.0.1.1', intf='sta1-wlan0')
    sta1.setIP('10.0.1.2', intf='sta1-wlan1')
    
    print("🔧 连接WiFi...")
    sta1.cmd('iw dev sta1-wlan0 connect ssid1')
    time.sleep(5)
    sta1.cmd('iw dev sta1-wlan1 connect ssid2')
    time.sleep(5)
    
    # 🎯 关键修复：配置策略路由
    print("🎯 配置路由表...")
    
    # 删除默认路由
    sta1.cmd('ip route del default 2>/dev/null || true')
    
    # 策略路由：基于源IP选择接口
    sta1.cmd('ip route add 10.0.1.10/32 dev sta1-wlan0 src 10.0.1.1 table 1')
    sta1.cmd('ip rule add from 10.0.1.1 table 1')
    sta1.cmd('ip route add 10.0.1.10/32 dev sta1-wlan1 src 10.0.1.2 table 2')
    sta1.cmd('ip rule add from 10.0.1.2 table 2')
    
    # 添加默认路由
    sta1.cmd('ip route add default dev sta1-wlan0')
    
    # 验证连接
    print("📊 验证连接状态:")
    wlan0_check = sta1.cmd("iwconfig sta1-wlan0 | grep ESSID")
    wlan1_check = sta1.cmd("iwconfig sta1-wlan1 | grep ESSID")
    print(f"WLAN0: {wlan0_check.strip()}")
    print(f"WLAN1: {wlan1_check.strip()}")
    
    net.pingAll()
    
    print("\n✅ 网络创建完成!")
    print("💡 现在可以用以下命令测试:")
    print("   sta1 iperf3 -c 10.0.1.10 -B 10.0.1.1 -t 20  # 使用wlan0")
    print("   sta1 iperf3 -c 10.0.1.10 -B 10.0.1.2 -t 20  # 使用wlan1")
    print("   用 ifstat -i sta1-wlan0,sta1-wlan1 监控流量")
    
    return net

if __name__ == "__main__":
    net = create_dual_wifi_network()
    CLI(net)
    net.stop()
