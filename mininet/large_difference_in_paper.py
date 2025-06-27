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
    check_for_real_interfaces_in_code()
    
    # 启用MPTCP（如果系统支持）
    print("尝试启用MPTCP...")
    os.system('echo 1 > /proc/sys/net/mptcp/mptcp_enabled 2>/dev/null || echo "MPTCP未安装，继续使用常规TCP"')
    
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

    # === 论文中的极端异构场景配置 ===
    print("=== 配置论文中的极端异构场景 ===")
    sta1.cmd('tc qdisc del dev sta1-eth0 root || true')
    sta1.cmd('tc qdisc del dev sta1-eth1 root || true')
    
    # WiFi路径: 8Mbps, 50ms RTT (论文原始参数)
    print("sta1-eth0: WiFi路径 - 8Mbps, 50ms RTT")
    sta1.cmd('tc qdisc add dev sta1-eth0 root handle 1:0 tbf rate 8mbit latency 50ms burst 1540')
    sta1.cmd('tc qdisc add dev sta1-eth0 parent 1:0 handle 10:0 netem delay 25ms')
    
    # LTE路径: 16Mbps, 270ms RTT (论文中最大差距场景)
    print("sta1-eth1: LTE路径 - 16Mbps, 270ms RTT")
    sta1.cmd('tc qdisc add dev sta1-eth1 root handle 1:0 tbf rate 16mbit latency 100ms burst 1540')
    sta1.cmd('tc qdisc add dev sta1-eth1 parent 1:0 handle 10:0 netem delay 135ms')

    print("=== 极端异构环境配置完成 ===")
    print("WiFi路径(eth0): 8Mbps,  50ms RTT")
    print("LTE路径(eth1):  16Mbps, 270ms RTT") 
    print("异构比例: 带宽2:1, 延迟5.4:1 (论文最大差距场景)")
    print("")
    
    print("=== 基础测试命令 ===")
    print("连通性测试：")
    print("  sta1 ping -I 10.0.1.1 10.0.1.10  # WiFi路径")
    print("  sta1 ping -I 10.0.1.2 10.0.1.10  # LTE路径")
    print("")
    print("单路径性能：")
    print("  h1 iperf3 -s &")
    print("  sta1 iperf3 -c 10.0.1.10 -B 10.0.1.1 -t 10  # WiFi")
    print("  sta1 iperf3 -c 10.0.1.10 -B 10.0.1.2 -t 10  # LTE")
    print("")
    print("=== MPTCP调度器测试 ===")
    print("查看当前调度器：")
    print("  cat /proc/sys/net/mptcp/mptcp_scheduler")
    print("")
    print("切换调度器（可在运行时切换）：")
    print("  echo 'default' > /proc/sys/net/mptcp/mptcp_scheduler")
    print("  echo 'roundrobin' > /proc/sys/net/mptcp/mptcp_scheduler")
    print("")
    print("MPTCP性能测试：")
    print("  h1 iperf3 -s &")
    print("  sta1 iperf3 -c 10.0.1.10 -t 30  # 会自动使用MPTCP")
    print("")
    print("文件下载测试：")
    print("  h1 dd if=/dev/zero of=/tmp/256MB bs=1M count=256")
    print("  h1 cd /tmp && python3 -m http.server 8080 &")
    print("  sta1 time wget -O /dev/null http://10.0.1.10:8080/256MB")
    print("")
    print("=== 监控命令 ===")
    print("流量监控：")
    print("  sta1 ifstat -i sta1-eth0,sta1-eth1 1")
    print("  sta1 watch -n1 'cat /proc/net/dev | grep sta1'")
    print("")
    print("进入节点shell：")
    print("  sudo mnexec -a $(pgrep -f 'sta1') bash")
    print("  sudo mnexec -a $(pgrep -f 'h1') bash")
    print("")
    
    # 预创建测试文件
    print("正在创建测试文件...")
    h1.cmd('dd if=/dev/zero of=/tmp/256MB bs=1M count=256 2>/dev/null &')
    print("256MB测试文件创建中（后台进行）...")
    
    print("=== 调度器切换说明 ===")
    print("✓ 可以在Mininet运行时动态切换调度器")
    print("✓ 不需要重新创建网络")
    print("✓ 切换后立即生效，适合对比测试")

    CLI(net)
    net.stop()

if __name__ == '__main__':
    simple_dual_link_shared_switch()