#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import Link
from mininet.cli import CLI
from mininet.log import setLogLevel
import os

def simple_dual_link_shared_switch():
    os.system('mn -c')
    setLogLevel('info')

    # 不再使用 TCLink，避免隐式 delay 设置
    net = Mininet(controller=Controller, link=Link, switch=OVSSwitch)

    print("创建节点")
    h1 = net.addHost('h1', ip='10.0.1.10')
    sta1 = net.addHost('sta1')
    s1 = net.addSwitch('s1')
    c0 = net.addController('c0')

    print("创建链路")
    net.addLink(sta1, s1, intfName1='sta1-eth0')
    net.addLink(sta1, s1, intfName1='sta1-eth1')
    net.addLink(h1, s1)

    print("启动网络")
    net.build()
    c0.start()
    s1.start([c0])

    print("配置 IP 和路由")
    sta1.setIP('10.0.1.1/24', intf='sta1-eth0')
    sta1.setIP('10.0.1.2/24', intf='sta1-eth1')

    # 设置策略路由
    sta1.cmd('ip rule add from 10.0.1.1 table 1')
    sta1.cmd('ip rule add from 10.0.1.2 table 2')
    sta1.cmd('ip route add 10.0.1.10/32 dev sta1-eth0 table 1')
    sta1.cmd('ip route add 10.0.1.10/32 dev sta1-eth1 table 2')
    sta1.cmd('ip route add default dev sta1-eth0')

    # 手动设置 tc 延迟（你可以根据需求改）
    sta1.cmd('tc qdisc add dev sta1-eth0 root netem delay 5ms')
    sta1.cmd('tc qdisc add dev sta1-eth1 root netem delay 20ms')

    print("手动 tc 设置完成，准备测试")
    print("示例测试：")
    print("sta1 ping -I 10.0.1.1 10.0.1.10")
    print("sta1 iperf3 -c 10.0.1.10 -B 10.0.1.1")
    print("sta1 ping -I 10.0.1.2 10.0.1.10")
    print("sta1 iperf3 -c 10.0.1.10 -B 10.0.1.2")

    CLI(net)
    net.stop()

if __name__ == '__main__':
    simple_dual_link_shared_switch()
