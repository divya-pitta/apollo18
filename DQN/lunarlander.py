"""
Deep Q network,

LunarLander-v2 example

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
import numpy as np
from gym import wrappers
import pickle
import matplotlib.pyplot as plt
# from DQN_Vanilla import DeepQNetwork
from DQN import DeepQNetwork
from DoubleDQN import DoubleDQN
from PrioritizedReplayDQN import PrioritizedReplayDQN
from DoublePriorDQN import DoublePrioritizedReplayDQN
import tensorflow as tf
env = gym.make('LunarLander-v2')
# env = env.unwrapped
env.seed(1)

#Network Type 
DQN = "DQN"
DOUBLE_DQN = "Double_DQN"
PRIORITIZED_DQN = "PrioritizedReplayDQN"
DOUBLE_PRIORITIZED_DQN = "DoublePrioritizedReplayDQN"

N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 50000
TARGET_REP_ITER = 4000
MAX_EPISODES = 3000
E_GREEDY = 0.95
E_INCREMENT = 0.00001
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 32
HIDDEN = [400, 400]
HIDDEN_UNITS = [200, 200]
RENDER = False

# runNetworkTypes = [DQN, DOUBLE_DQN, PRIORITIZED_DQN]
runNetworkTypes = [DOUBLE_PRIORITIZED_DQN]

for NETWORK_TYPE in runNetworkTypes: 
    tf.reset_default_graph()
    print("---------------------------------------------------------------------------------------------------------")
    print(NETWORK_TYPE)
    if(NETWORK_TYPE == DQN):
        RL = DeepQNetwork(n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
            batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER, memory_size=MEMORY_CAPACITY, 
            e_greedy_increment=E_INCREMENT, output_graph=False, hiddenunits=HIDDEN_UNITS)

    elif(NETWORK_TYPE == DOUBLE_DQN):
        RL = DoubleDQN(n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
            batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER, memory_size=MEMORY_CAPACITY,
            e_greedy_increment=E_INCREMENT, output_graph=False, hiddenunits=HIDDEN_UNITS)

    elif(NETWORK_TYPE == PRIORITIZED_DQN):
        RL = PrioritizedReplayDQN(n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
            batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,memory_size=MEMORY_CAPACITY, 
            e_greedy_increment=E_INCREMENT, output_graph=True, hiddenunits=HIDDEN_UNITS)

    elif(NETWORK_TYPE == DOUBLE_PRIORITIZED_DQN):
        RL = DoublePrioritizedReplayDQN(n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
            batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,memory_size=MEMORY_CAPACITY, 
            e_greedy_increment=E_INCREMENT, output_graph=True, hiddenunits=HIDDEN_UNITS)

    total_steps = 0
    running_r = 0
    r_scale = 1
    landedcount = 0
    startStateValueList = []
    runningRewardList = []
    episodeRewardList = []
    landingList = []

    for i_episode in range(MAX_EPISODES):
        s = env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
        ep_r = 0
        ep_qval = 0
        ep_val = 0
        _, val = RL.choose_action(s)
        startStateValueList.append(val)
        while True:
            if total_steps > MEMORY_CAPACITY and RENDER: env.render()
            a, val = RL.choose_action(s)
            s_, r, done, _ = env.step(a)
            if r == -100: r = -30
            r /= r_scale

            ep_r += r
            ep_val += val
            RL.store_transition(s, a, r, s_)
            if total_steps > MEMORY_CAPACITY:
                # print("Learning taking place...............")
                RL.learn()
            # RL.learn()
            # if(total_steps>=500):
            #     exit()
            if done:
                land = '| Landed' if r == 100/r_scale else '| ------'
                if(land == '| Landed' or r == 100/r):
                    landedcount += 1
                    landingList.append(1)
                    # if(landedcount>10):
                    #     RENDER = True
                else:
                    landingList.append(0)

                running_r = 0.99 * running_r + 0.01 * ep_r
                runningRewardList.append(running_r)
                episodeRewardList.append(ep_r)
                if(i_episode%50 == 0):
                    print('Epi: ', i_episode,
                          land,
                          '| Epi_R: ', round(ep_r, 2),
                          '| Running_R: ', round(running_r, 2),
                          '| Epi_Val: ', round(ep_val, 2),
                          '| Epi_qEval: ', round(ep_qval, 2),
                          '| Epsilon: ', round(RL.epsilon, 3))
                break

            s = s_
            total_steps += 1

    # RL.plot_cost()

    def plot_cost(pltList, xlabel, ylabel, title):
        plt.figure()
        plt.plot(np.arange(len(pltList)), pltList)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
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