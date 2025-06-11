#!/bin/bash
NS_NAME="ns-mptcp"
# Path to the Python sender script
#SCRIPT_PATH="/home/vagrant/mptcp-rl-scheduler/src/reles/server.py"
SCRIPT_PATH="/home/vagrant/mptcp-rl-scheduler/src/reles/sender.py"
sudo ip netns exec $NS_NAME python3 $SCRIPT_PATH 1 default
#sudo ip netns exec ns-mptcp ifstat -i uesimtun0,gretun-id-2-1
