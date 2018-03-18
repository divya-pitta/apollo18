import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np

# def movingaverage(interval, window_size):
#     window = numpy.ones(int(window_size))/float(window_size)
#     return numpy.convolve(interval, window, 'same')
def slidingavg(pltList, window_width):
	cumsum_vec = np.cumsum(np.insert(pltList, 0, 0)) 
	pltList = (cumsum_vec[window_width:] - cumsum_vec[:-window_width])/window_width
	return pltList

def plot_cost(pltList, window_width, xlabel, ylabel, title):
	pltList = slidingavg(pltList, window_width)
	plt.figure()
	plt.plot(np.arange(len(pltList)), pltList)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.savefig("{0}_{1}".format(title,ylabel))

def plot_cost_multiple(pltList, window_width, xlabel, ylabel, title, labelList):
	plt.figure()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	for i in range(len(pltList)):
		pltList[i] = slidingavg(pltList[i], window_width)
		plt.plot(np.arange(len(pltList[i])), pltList[i], label=labelList[i])
	plt.legend()
	print(title)
	plt.savefig("{0}".format(title))


rootFolder = sys.path[0]+'\\Exp6\\'
# rootFolderNames = ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]
rootFolderNames = ["Exp2", "Exp6"]
rootFolderList  = [sys.path[0]+"\\"+x+"\\" for x in rootFolderNames]

dqn = "DQN"
doubleDQN = "Double_DQN"
prioritizedReplay = "PrioritizedReplayDQN"
DOUBLE_PRIORITIZED_DQN = "DoublePrioritizedReplayDQN"

landing = "Landing"
runningReward = "RunningReward"
startStateValue = "StartStateValue"
episodeReward = "EpisodeReward"

plotNetworks = [dqn, doubleDQN, prioritizedReplay, DOUBLE_PRIORITIZED_DQN]
plotsList = [landing, runningReward, startStateValue, episodeReward]

# for nwType in plotNetworks:
# 	for p in plotsList:
# 		listInfo = pickle.load(open(rootFolder+nwType+"_"+p+".pb","rb"))
# 		plot_cost(listInfo, 50, "Episode", p, nwType)

# for p in plotsList:
# 	pltList = []
# 	labelList = []
# 	for nwType in plotNetworks:
# 		listInfo = pickle.load(open(rootFolder+nwType+"_"+p+".pb", "rb"))
# 		pltList.append(listInfo)
# 		labelList.append(nwType)
# 	plot_cost_multiple(pltList, 50, "Episode", p, "Comparison of "+p, labelList)

for p in plotsList:
	for nwType in plotNetworks:
		pltList = []
		labelList = []
		i=0
		for folder in rootFolderList:
			try:
				listInfo = pickle.load(open(folder+nwType+"_"+p+".pb", "rb"))
				pltList.append(listInfo)
				labelList.append(nwType+"_"+rootFolderNames[i])
			except:
				doNothing = True
			i += 1
		plot_cost_multiple(pltList, 50, "Episode", p, "Comparison of "+p+" for network "+nwType, labelList)