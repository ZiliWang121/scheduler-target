#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import time
import os
from datetime import datetime
import seaborn as sns
import pandas as pd
import statistics 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools
from statannotations.Annotator import Annotator
import matplotlib


if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser(description='Give scheduler name for debug,test or training')
	parser.add_argument('--schedulers',type=int,default=126,help='Name of the schedulers combination to evaluate')
	parser.add_argument("--scenario",type=str,default="LiFi_Ethernet",help="choose static(0) or dynamic(1) links")
	parser.add_argument("--scenarios",action="store_true",help="Comparing different scenarios")
	parser.add_argument("--files",action="store_true",help="Comparing performance of schedulers for file sizes")
	args = parser.parse_args()
	plot_fsize=13
	TCP_only = (("TCP","Ethernet"),)
	base_schedulers = (("default","default"),("blest","blest"))
	RL_schedulers = (("reles","reles"),("falcon","falcon"),('reles_ext','reles_e'),("falcon_ext","falcon_e"))
	links = ["Ethernet","WiFi 6","LiFi"] #,"WLAN"]
	
	schedulers = np.concatenate((TCP_only,base_schedulers,RL_schedulers))
	all_combinations = []
	xticklabel1 = []
	xticklabel2 = []
	
	for L in range(len(schedulers)+1):
		for subset in itertools.combinations(schedulers,L):
			all_combinations.append(subset)
	print(all_combinations[args.schedulers])
	schedulers = all_combinations[args.schedulers] #all_combinations[63] #args.schedulers]
	scenario = args.scenario  #e.g LiFi_Ethernet or LiFi_WiFi
	#file names = scenario + _scheduler + _filesize
	#file name = scheduler_ + LiFi_Ethernet2mb (scenario)  -> e.g default_LiFi_Ethernet2mb
	
	fig = plt.figure(1,figsize=(5,9))
	##if scenario.find("dynamic") != -1:
		#gs = GridSpec(5,3,figure=fig)
#	else:
	gs = GridSpec(2,1,figure=fig)
	ax1 = fig.add_subplot(gs[0,:])
	ax2 = fig.add_subplot(gs[1,:])
	all_data = pd.DataFrame()
	for i in range(len(schedulers)):
		print(schedulers[i][0]+scenario+".csv")
		csv = pd.read_csv(schedulers[i][0]+"_"+scenario+".csv")
		#ax1.boxplot(csv["completion time"].dropna(),positions=[i],showfliers=False)
		all_data = pd.concat([all_data,csv["completion time"].dropna().rename("comp"+str(i))],axis=1)
		ax2.plot(csv["throughput"].dropna())
		xticklabel1.append(schedulers[i][1])
	
	idx = np.arange(start=0,stop=len(xticklabel1))
	
	axa = sns.boxplot(data=all_data,ax=ax1,showfliers=False,width=0.35)
	pairs = []
	for k in range(1,len(schedulers)):
		pairs.append(("comp0",all_data.columns[k]))
		
	pairs.append(("comp2","comp4"))
	pairs.append(("comp3","comp5"))
	print(pairs)
	annotator = Annotator(axa,pairs,data=all_data)
	annotator.configure(test="Mann-Whitney-gt",comparisons_correction=None,text_format="full",
	line_width=0.85,text_offset=0.85,fontsize="x-small")
	annotator.apply_and_annotate()
	
	ax1.set_ylabel("Completion Time in s",fontsize=plot_fsize)
	ax1.set_xticklabels(xticklabel1)
	ax1.set_axisbelow(True)
	ax1.grid()
	ax2.set_ylabel("Throughput in MB/s",fontsize=plot_fsize)
	ax2.legend(xticklabel1) 
	ax2.set_xlabel("File Transfer",fontsize=plot_fsize)
	ax2.set_axisbelow(True)
	ax2.grid()
	fig.savefig("../../../test/old_plots/hardware_evaluation1"+scenario+(time.ctime()),dpi=400,bbox_inches="tight")
	#fig.savefig("../../../test/old_plots/hardware_evaluation1"+scenario+(time.ctime())+".pdf",dpi=100,bbox_inches="tight")
	manager = plt.get_current_fig_manager()
	manager.resize(*manager.window.maxsize())
	plt.show()
	
	if args.scenarios:
		boxplots = []
		colors = ["lightgreen","lightblue","pink","orange","black","purple","lightgreen","lightblue","pink","orange","black","purple"]
		fig = plt.figure(figsize=(6,4.5))
		scenarios = ["LiFi_Ethernet2mb","LiFi_WiFi2mb"]
		for j in range(len(scenarios)):
			for i in range(len(schedulers)):
				print(schedulers[i][0]+"_"+scenarios[j]+".csv")
				csv = pd.read_csv(schedulers[i][0]+"_"+scenarios[j]+".csv")
				boxplots.append(plt.boxplot(csv["completion time"].dropna(),positions=[j+(i*0.1-0.25)],widths=0.1,
				showfliers=False,patch_artist=True,zorder=3))
				xticklabel2.append(scenarios[j])
				
		idx = np.arange(start=0,stop=len(scenarios))
		

		for i in range(len(boxplots)):
			for patch in boxplots[i]["boxes"]:
				patch.set_facecolor(colors[i])
				patch.set_alpha(0.65)
		
		plt.legend([boxplots[0]["boxes"][0],boxplots[1]["boxes"][0],boxplots[2]["boxes"][0],
		boxplots[3]["boxes"][0],boxplots[4]["boxes"][0],boxplots[5]["boxes"][0]],xticklabel1)
		plt.ylabel("Completion Time in s",fontsize=plot_fsize)
		plt.xlabel("Link Combination",fontsize=plot_fsize)
		plt.xticks(idx,scenarios,fontsize=plot_fsize)
		plt.grid(zorder=0)
		plt.show()
		fig.savefig("../../../test/old_plots/hardware_evaluation_all"+scenario+(time.ctime()),dpi=400,bbox_inches="tight")
		#fig.savefig("../../../test/old_plots/hardware_evaluation_all"+scenario+(time.ctime())+".pdf",dpi=100,bbox_inches="tight")
	
	if args.files:
		fig = plt.figure(figsize=(6,4.5))
		file_sizes = ["64kb","2mb","8mb","64mb"]
		colors = ["green","blue","red","orange","black","purple"]
		for j in range(len(file_sizes)):
			scenario_ = scenario.replace("2mb",file_sizes[j])
			csvv = pd.read_csv(schedulers[0][0]+"_"+scenario_+".csv")
			normalizationv = csvv["completion time"].dropna().mean()
			for i in range(len(schedulers)):
				print(schedulers[i][0]+"_"+scenario_+".csv")
				csv = pd.read_csv(schedulers[i][0]+"_"+scenario_+".csv")
				#normalize completion time to first scheduler of combination
				norm_ct = csv["completion time"].dropna().mean()/normalizationv
				plt.bar([j+(0.075*(i-2))],norm_ct,width=0.075,edgecolor=colors[i],hatch='/',alpha=0.8,fill=False,zorder=3)
		
		idx = np.arange(start=0,stop=len(file_sizes))
		plt.legend(xticklabel1,loc="lower center")
		plt.ylabel("Normalized Completion Time",fontsize=plot_fsize)
		plt.xlabel("Download sizes",fontsize=plot_fsize)
		plt.xticks(idx,file_sizes)
		plt.grid(zorder=0)
		plt.savefig("../../../test/old_plots/hardware_evaluation_files"+scenario+(time.ctime()),dpi=400,bbox_inches="tight")
		#plt.savefig("../../../test/old_plots/hardware_evaluation_files"+scenario+(time.ctime())+".pdf",dpi=100,bbox_inches="tight")
		manager = plt.get_current_fig_manager()
		manager.resize(*manager.window.maxsize())
		plt.show()
