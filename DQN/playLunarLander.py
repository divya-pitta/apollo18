import gym
import numpy as np
from gym import wrappers
import pickle
import matplotlib.pyplot as plt
from d_dqn import DQN
import tensorflow as tf
env = gym.make('LunarLander-v2')
env.seed(1)

#Network Type 
sDQN = "DQN"
sDOUBLE_DQN = "Double_DQN"
sPRIORITIZED_DQN = "PrioritizedReplayDQN"
sDOUBLE_PRIORITIZED_DQN = "DoublePrioritizedReplayDQN"

ACTION_DIM = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
EPS_MIN = 0.5
EPS_MAX = 0.95
EPS_STEP = 0.00001
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 32
TUPLE_MEMORY_COUNT = 50000
REPLACE_TARGET_IN = 4000
EPISODES_MAX = 3000
HIDDEN_LAYERS = 2
HIDDEN_UNITS = 100

runNetworkTypes = [sDOUBLE_DQN, sPRIORITIZED_DQN, sDOUBLE_PRIORITIZED_DQN, sDQN]
RENDER = False

# runNetworkTypes = [DOUBLE_PRIORITIZED_DQN]

for NETWORK_TYPE in runNetworkTypes: 
    tf.reset_default_graph()
    print("---------------------------------------------------------------------------------------------------------")
    print(NETWORK_TYPE)
    RL = DQN(statedim = STATE_DIM, actiondim = ACTION_DIM, gamma = GAMMA, 
            epsmin = EPS_MIN, epsmax = EPS_MAX, epsstep = EPS_STEP, 
            tupleMemoryCount = TUPLE_MEMORY_COUNT, tupleBatchSize= BATCH_SIZE, 
            replacetargetineps = REPLACE_TARGET_IN, optimizer = tf.train.AdamOptimizer, 
            lr = LR, hiddenlayers = HIDDEN_LAYERS, hiddenunits = HIDDEN_UNITS, dqnType = NETWORK_TYPE)

    total_steps = 0
    running_r = 0
    r_scale = 1
    landedcount = 0
    startStateValueList = []
    runningRewardList = []
    episodeRewardList = []
    landingList = []

    for i_episode in range(EPISODES_MAX):
        s = env.reset()
        ep_r = 0
        ep_qval = 0
        ep_val = 0
        _, val = RL.selectAction(s)
        startStateValueList.append(val)
        while True:
            if total_steps > TUPLE_MEMORY_COUNT and RENDER: env.render()
            a, val = RL.selectAction(s)
            s_, r, done, _ = env.step(a)
            if r == -100: r = -30

            ep_r += r
            ep_val += val
            RL.addTupleToMemory(s, a, r, s_)
            if total_steps > TUPLE_MEMORY_COUNT:
                RL.backprop()
            # RL.backprop()
            if done:
                land = '| Landed' if r == 100 else '| ------'
                if(land == '| Landed' or r == 100):
                    landedcount += 1
                    landingList.append(1)
                else:
                    landingList.append(0)
                running_r = 0.99 * running_r + 0.01 * ep_r
                runningRewardList.append(running_r)
                episodeRewardList.append(ep_r)
                if(i_episode%500 == 0):
                    print('Epi: ', i_episode,
                          land,
                          '| Epi_R: ', round(ep_r, 2),
                          '| Running_R: ', round(running_r, 2),
                          '| Epi_Val: ', round(ep_val, 2),
                          '| Epi_qEval: ', round(ep_qval, 2),
                          '| Epsilon: ', round(RL.eps, 3))
                break
            s = s_
            total_steps += 1

    def plot_cost(pltList, xlabel, ylabel, title):
        plt.figure()
        plt.plot(np.arange(len(pltList)), pltList)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        print(title)
        # plt.show()
        plt.savefig("{0}_{1}".format(title,ylabel))

    plot_cost(startStateValueList, "Episode", "Value of start Episode", NETWORK_TYPE)
    plot_cost(episodeRewardList, "Episode", "Reward per episode", NETWORK_TYPE)
    plot_cost(runningRewardList, "Episode", "Running per episode reward", NETWORK_TYPE)
    plot_cost(landingList, "Episode", "Landing", NETWORK_TYPE)
    pickle.dump(startStateValueList, open("{0}_{1}.pb".format(NETWORK_TYPE, "StartStateValue"), "wb"))
    pickle.dump(episodeRewardList, open("{0}_{1}.pb".format(NETWORK_TYPE, "EpisodeReward"), "wb"))
    pickle.dump(runningRewardList, open("{0}_{1}.pb".format(NETWORK_TYPE, "RunningReward"), "wb"))
    pickle.dump(landingList, open("{0}_{1}.pb".format(NETWORK_TYPE, "Landing"), "wb"))

    del(RL)