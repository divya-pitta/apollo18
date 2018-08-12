import gym
from PIL import Image
import pickle
import pandas as pd
from ggplot import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

env = gym.make('LunarLander-v2')
env.reset()
frames = []
# while True:
#     frames.append(Image.fromarray(env.render(mode='rgb_array')))  # save each frames
#     env.env.ale.saveScreenPNG('test_image.png')
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if done:
#         if reward:
#             print(reward)
#         break
		
# with open('openai_gym.gif', 'wb') as f:  # change the path if necessary
#     im = Image.new('RGB', frames[0].size)
#     # im.save(f)
#     im.save(f, save_all=True, append_images=frames)

stateActionList = pickle.load(open('StateActionListPlotData.pb', 'rb'))
stateList = stateActionList[0]
actionList = stateActionList[1]
episodeCount = stateActionList[2]
print(len(stateList))
print(len(actionList))
print(len(episodeCount))

# tsne = TSNE(n_components=2, verbose=1)
# lowDimStateSpace = tsne.fit_transform(stateList)
# pickle.dump(lowDimStateSpace, open('LowDimStateSpace.pb', 'wb'))
lowDimStateSpace = pickle.load(open('LowDimStateSpace.pb', 'rb'))
print(episodeCount)
lowDimStateSpace = lowDimStateSpace[:330]
actionList = actionList[:330]

stateActListX = [[], [], [], []]
stateActListY = [[], [], [], []]
posnMapper = [[], [], [], []]

for i in range(len(actionList)):
	if(actionList[i]==0):
		stateActListX[0].append(lowDimStateSpace[i][0])
		stateActListY[0].append(lowDimStateSpace[i][1])
		posnMapper[0].append(i)
	elif(actionList[i]==1):
		stateActListX[1].append(lowDimStateSpace[i][0])
		stateActListY[1].append(lowDimStateSpace[i][1])
		posnMapper[1].append(i)
	elif(actionList[i]==2):
		stateActListX[2].append(lowDimStateSpace[i][0])
		stateActListY[2].append(lowDimStateSpace[i][1])
		posnMapper[2].append(i)
	elif(actionList[i]==3):
		stateActListX[3].append(lowDimStateSpace[i][0])
		stateActListY[3].append(lowDimStateSpace[i][1])
		posnMapper[3].append(i)
# print(len(stateActListX[2]))
# for i in range(331,len(stateList)):
# 	os.rename('./FrameImages/{}.png'.format(str(i+1)), './FrameImages/{}_{}_{}.png'.format(str(i+1), 
# 		str(lowDimStateSpace[i][0]), str(lowDimStateSpace[i][1])))

plt.scatter(stateActListX[0], stateActListY[0], color='red', label='Do Nothing')
plt.scatter(stateActListX[1], stateActListY[1], color='green', label='Fire Left Engine')
plt.scatter(stateActListX[2], stateActListY[2], color='blue', label='Fire Main Engine')
plt.scatter(stateActListX[3], stateActListY[3], color='yellow', label='Fire Right Engine')
for i in range(4):
	for j in range(len(stateActListX[i])):
		xy=(stateActListX[i][j],stateActListY[i][j])
		plt.annotate(posnMapper[i][j],xy)
plt.legend()

plt.show()


print("Done transforming data")

# stateFrame = pd.DataFrame(np.column_stack([lowDimStateSpace[:,0], lowDimStateSpace[:,1], actionList]),
# 	columns=['state1', 'state2', 'action'])

# chart = ggplot( stateFrame, aes(x='state1', y='state2', color='action') ) \
#         + geom_point(size=75,alpha=0.8) \
#         + scale_color_brewer(type='seq', palette='Reds') \
#         + ggtitle("State Action Plot")
# print(chart)

