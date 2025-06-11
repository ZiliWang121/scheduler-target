#!/usr/bin/python3

import sys
import os
from os import path
import time
import threading
import pickle
from threading import Event
import socketserver
import numpy as np
import socket
import mpsched
import torch
from configparser import ConfigParser
from replay_memory import ReplayMemory
from agent import Online_Agent, Offline_Agent
from naf_lstm import NAF_LSTM
from gym import spaces
import http.server
import multiprocessing
from datetime import datetime
import shutil

#structure and modulisation based on github.com/gaogogo/Experiment


class MyHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with overwritten do_GET function to give information about start of file transfer
    and the socket fd to the online agent 
    """
    def do_GET(self):

        sock= self.request
        agent = Online_Agent(fd=sock.fileno(),cfg=self.server.cfg,memory=self.server.replay_memory,event=self.server.event)
        agent.start()
        self.server.event.set()
        f = self.send_head()
        if f:
            try:
                self.copyfile(f,self.wfile)
            finally:
                f.close()
                self.server.event.clear()

class ThreadedHTTPServer(socketserver.ThreadingMixIn,http.server.HTTPServer):
    """ThreadedHTTPServer class initialized with (IP,PORT),HTTPRequestHandler
    """
    pass

def main(argv):
    cfg = ConfigParser()
    cfg.read('config.ini')
    IP = cfg.get('server','ip')
    PORT = cfg.getint('server','port')
    MEMORY_FILE = cfg.get('replaymemory','memory')
    AGENT_FILE = cfg.get('nafcnn','agent')
    INTERVAL = cfg.getint('train','interval')
    EPISODE = cfg.getint('train','episode')
    BATCH_SIZE = cfg.getint('train','batch_size')
    MAX_NUM_FLOWS = cfg.getint("env",'max_num_subflows')
    transfer_event = Event()
    CONTINUE_TRAIN = 1

    if len(argv) != 0:
        CONTINUE_TRAIN = int(argv[0])
        now = datetime.now().replace(microsecond=0)
        start_train = now.strftime("%Y-%m-%d %H:%M:%S")
        scenario = argv[1]
    #print(argv)
    print(f"[Server] Starting with args CONTINUE_TRAIN={CONTINUE_TRAIN}, scenario='{scenario}'")
    if os.path.exists(MEMORY_FILE) and CONTINUE_TRAIN:
        with open(MEMORY_FILE,'rb') as f:
            try:
                memory = pickle.load(f)
                f.close()
            except EOFError:
                print("memory EOF error not saved properly")
                memory = ReplayMemory(cfg.getint('replaymemory','capacity'))
    else:
        memory = ReplayMemory(cfg.getint('replaymemory','capacity'))

    if CONTINUE_TRAIN != 1 and os.path.exists(AGENT_FILE):
        os.makedirs("trained_models/",exist_ok=True)
        shutil.move(AGENT_FILE,"trained_models/agent"+start_train+".pkl")
    if not os.path.exists(AGENT_FILE) or CONTINUE_TRAIN != 1:
        agent = NAF_LSTM(gamma=cfg.getfloat('nafcnn','gamma'),tau=cfg.getfloat('nafcnn','tau'),
        hidden_size=cfg.getint('nafcnn','hidden_size'),num_inputs=cfg.getint('env','k')*MAX_NUM_FLOWS*5,
        action_space=MAX_NUM_FLOWS) #5 is the size of state space (TP,RTT,CWND,unACK,retrans)
        torch.save(agent,AGENT_FILE)

    off_agent = Offline_Agent(cfg=cfg,model=AGENT_FILE,memory=memory,event=transfer_event)
    off_agent.daemon = True
    server = ThreadedHTTPServer((IP,PORT),MyHTTPHandler)
    server.event = transfer_event
    server.cfg = cfg
    server.replay_memory = memory
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    try:
        while(transfer_event.wait(timeout=60)): #only returns false in case of timeout
            if len(memory) > BATCH_SIZE and not off_agent.is_alive():
                off_agent.start() 
            time.sleep(25)
            pass
        with open(MEMORY_FILE,'wb') as f:
            pickle.dump(memory,f)
            f.close()
    except (KeyboardInterrupt,SystemExit):
        with open(MEMORY_FILE,'wb') as f:
            pickle.dump(memory,f)
            f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
    
    
    
    
        
        
