"""
We referred to https://morvanzhou.github.io/tutorials/. Our code is inspired by the tutorial version.
We made it suit our needs, incorporate some new algorithms and tune parameters easily.
"""


# RNN replaced with dense layer

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import random
import pickle

MODEL_NAME = "Hidden_128"
OUTPUT_DIR = "./LunarLanderLog2"

GAME = 'LunarLander-v2'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 5000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.001   # not useful in this case
LR_A = 0.0005    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
PER_EPISODE_REWARD = []
GLOBAL_EP = 0
NUM_HIDDEN_UNITS=8

# VALIDATION_STATES = []
VALIDATION_STATES = pickle.load(open("validation_states.p", "rb"))
NUM_VALIDATION_STATES = len(VALIDATION_STATES)
VAL_STATE_VALUE = [[] for i in range(NUM_VALIDATION_STATES)]

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.n
del env

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net(N_A)
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net(N_A)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(tf.clip_by_value(self.a_prob, 1e-20, 1.0)) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(tf.clip_by_value(self.a_prob, 1e-20, 1.0)), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, n_a):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('critic'):
            hidden_layer_size = NUM_HIDDEN_UNITS
            print("Hidden layer size = ", hidden_layer_size)
            print("self.s shape = ", self.s.shape)
            cell_out = tf.layers.dense(self.s, hidden_layer_size, tf.nn.relu6, kernel_initializer=w_init, name='hl')
            l_c = tf.layers.dense(cell_out, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        with tf.variable_scope('actor'):
            cell_out = tf.stop_gradient(cell_out, name='c_cell_out')
            l_a = tf.layers.dense(cell_out, 300, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, n_a, tf.nn.softmax, kernel_initializer=w_init, name='ap')

        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

         # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run([self.a_prob], feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(len(np.array(prob_weights).ravel())), p=np.array(prob_weights).ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        r_scale = 100
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            ep_t = 0
            while True:
                # if self.name == 'W_0' and total_step % 10 == 0:
                #     self.env.render()
                a = self.AC.choose_action(s)  # get the action
                s_, r, done, info = self.env.step(a)
                if r == -100: r = -10
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r/r_scale)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0,0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target
                    }
                    critic_loss, actor_loss = SESS.run([self.c_loss, self.a_loss], feed_dict=feed_dict)
                    print(critic_loss, " ", actor_loss)
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_

                # With probability 0.01 add this to the validation states
                # prob_validation = random.uniform(0, 1)
                # if (prob_validation <= 0.0001 and len(VALIDATION_STATES) < NUM_VALIDATION_STATES):
                #     print("Storing this state ", self.name , " ", GLOBAL_EP)
                #     VALIDATION_STATES.append(s)
                total_step += 1
                ep_t += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    # Evaluate the model on the validation state
                    # v_val_state = SESS.run(self.AC.v, {self.AC.s: validation_state[np.newaxis, :]})[0,0]
                    # VAL_STATE_VALUE.append(v_val_state)

                    for j in range(NUM_VALIDATION_STATES):
                        v_val_state = SESS.run(self.AC.v, {self.AC.s: VALIDATION_STATES[j][np.newaxis, :]})[0,0]
                        # print("v_val_state: ", v_val_state)
                        VAL_STATE_VALUE[j].append(v_val_state)
                    PER_EPISODE_REWARD.append(ep_r)
                    if not self.env.unwrapped.lander.awake: solve = '| Landed'
                    else: solve = '| ------'

                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        solve,
                        "| Running_Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        "| Ep_r: %.2f" % PER_EPISODE_REWARD[-1],
                        file = logfile
                        )
                    logfile.flush()
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    for batch_size in [5]:
        for NUM_HIDDEN_UNITS in [16, 32]:
            for workers in [4]:
                for beta in [0.001]:
                    ENTROPY_BETA = beta
                    UPDATE_GLOBAL_ITER = batch_size
                    N_WORKERS = workers
                    GLOBAL_EP = 0
                    GLOBAL_RUNNING_R = []
                    PER_EPISODE_REWARD = []
                    VAL_STATE_VALUE = [[] for i in range(NUM_VALIDATION_STATES)]

                    MODEL_NAME = "{0}_{1}_{2}".format()
                    tf.reset_default_graph()
                    logfile = open("{0}/log_{1}.txt".format(OUTPUT_DIR, MODEL_NAME), "w")
                    SESS = tf.Session()

                    with tf.device("/cpu:0"):
                        OPT_A = tf.train.AdamOptimizer(LR_A, name='AdamA')
                        OPT_C = tf.train.AdamOptimizer(LR_C, name='AdamC')
                        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
                        workers = []
                        # Create worker
                        for i in range(N_WORKERS):
                            i_name = 'W_%i' % i   # worker name
                            workers.append(Worker(i_name, GLOBAL_AC))

                    COORD = tf.train.Coordinator()
                    SESS.run(tf.global_variables_initializer())

                    if OUTPUT_GRAPH:
                        if os.path.exists(LOG_DIR):
                            shutil.rmtree(LOG_DIR)
                        tf.summary.FileWriter(LOG_DIR, SESS.graph)

                    worker_threads = []
                    for worker in workers:
                        job = lambda: worker.work()
                        t = threading.Thread(target=job)
                        t.start()
                        worker_threads.append(t)
                    COORD.join(worker_threads)

                    # print("Number of validation states: ", len(VALIDATION_STATES))
                    # print(VALIDATION_STATES)
                    # pickle.dump( VALIDATION_STATES, open( "validation_states.p", "wb" ) )

                    valstatefile = "{0}/val_s_value_{1}.p".format(OUTPUT_DIR, MODEL_NAME)
                    rewardfile = "{0}/reward_{1}.p".format(OUTPUT_DIR, MODEL_NAME)

                    pickle.dump(VAL_STATE_VALUE, open(valstatefile, "wb"))
                    pickle.dump([GLOBAL_RUNNING_R, PER_EPISODE_REWARD], open(rewardfile, "wb"))
                    logfile.close()

                    # for i in range(NUM_VALIDATION_STATES):
                    #     plt.subplot(NUM_VALIDATION_STATES/2+1, 2, i+1)
                    #     plt.plot(np.arange(len(VAL_STATE_VALUE[i])), VAL_STATE_VALUE[i])

                    # plt.subplot(NUM_VALIDATION_STATES/2+1, 2, NUM_VALIDATION_STATES+1)
                    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
                    # plt.xlabel('step')
                    # plt.ylabel('Total moving reward')
                    # plt.show()
