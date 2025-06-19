#!/usr/bin/env python

from mininet.net import Mininet
from mininet.node import Controller, OVSSwitch
from mininet.link import Link
from mininet.cli import CLI
from mininet.log import setLogLevel
import os
import sys

# 替换原来的 check_for_real_interfaces() 内容为：
def check_for_real_interfaces_in_code():
    print("正在检查代码中是否误用了真实网卡名...")

    # 可疑接口名列表（只报警如果代码中引用了它们）
    dangerous_ifaces = ['eno1', 'enp2s0f0', 'enp2s0f1']

    # 当前脚本名
    this_file = os.path.abspath(__file__)

    # 逐个接口名检查代码中是否引用
    for iface in dangerous_ifaces:
        grep_cmd = f"grep -w '{iface}' {this_file}"
        result = os.popen(grep_cmd).read()
        if iface in result:
            print(f"警告：在脚本代码中发现引用真实网卡名 {iface}，已退出以保护系统！")
            sys.exit(1)

    print("脚本中未引用任何真实网卡，运行安全。")


def simple_dual_link_shared_switch():
    os.system('mn -c')
    setLogLevel('info')

    # 安全检测
    check_for_real_interfaces_in_code
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
    sta1.cmd('tc qdisc del dev sta1-eth0 root || true')
    sta1.cmd('tc qdisc del dev sta1-eth1 root || true')
    # sta1-eth0: 高带宽低延迟
    sta1.cmd('tc qdisc add dev sta1-eth0 root handle 1: tbf rate 100mbit burst 15k latency 50ms')
    sta1.cmd('tc qdisc add dev sta1-eth0 parent 1:1 handle 10: netem delay 20ms')
    # sta1-eth1: 低带宽高延迟
    sta1.cmd('tc qdisc add dev sta1-eth1 root handle 1: tbf rate 20mbit burst 15k latency 50ms')
    sta1.cmd('tc qdisc add dev sta1-eth1 parent 1:1 handle 10: netem delay 20ms')

    print("手动 tc 设置完成，准备测试")
    print("示例测试：")
    print("sta1 ping -I 10.0.1.1 10.0.1.10")
    print("sta1 iperf3 -c 10.0.1.10 -B 10.0.1.1")
    print("sta1 ping -I 10.0.1.2 10.0.1.10")
    print("sta1 iperf3 -c 10.0.1.10 -B 10.0.1.2")
    print("进入两个节点：")
    print("sudo mnexec -a $(pgrep -f 'sta1') bash")
    print("sudo mnexec -a $(pgrep -f 'h1') bash")
    print("流量监测：")
    print("进入两个节点：")
    print("ifstat -i sta1-eth0,sta1-eth1")
    

    CLI(net)
    net.stop()

if __name__ == '__main__':
    simple_dual_link_shared_switch()
