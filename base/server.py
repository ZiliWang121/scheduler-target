#!/usr/bin/python3

#structure and modulisation based on github.com/gaogogo/Experiment

import os
import sys
import time
import threading
from threading import Event
import pickle
import socketserver
import numpy as np
import socket
import mpsched
from configparser import ConfigParser
import http.server



class MyHTTPHandler(http.server.SimpleHTTPRequestHandler):
	def do_GET(self):
		
		self.server.event.set()
		sock= self.request
		print(sock.fileno())
		print(mpsched.get_sub_info(sock.fileno()))
		f = self.send_head()
		if f:
			try:
				self.copyfile(f,self.wfile)
			finally:
				f.close()
		self.server.event.clear()
		
		
class ThreadedHTTPServer(socketserver.ThreadingMixIn,http.server.HTTPServer):
	pass

def main(argv):
	cfg = ConfigParser()
	cfg.read('config.ini')
	IP = cfg.get('server','ip')
	PORT = cfg.getint('server','port')
	transfer_event = Event()
	replay_memory  = 0
	
	print(argv)
	if len(argv) != 0:
		print("continue")
	server = ThreadedHTTPServer((IP,PORT),MyHTTPHandler)
	server.event = transfer_event
	server_thread = threading.Thread(target=server.serve_forever)
	server_thread.daemon = True
	server_thread.start()	
	while(transfer_event.wait(timeout=60)): 
		time.sleep(25)
		pass
	

if __name__ == '__main__':
	main(sys.argv[1:])
	
	
	
	
	
	
		
		
