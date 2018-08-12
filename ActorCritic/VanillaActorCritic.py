import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

global GLOBAL_RUNNING_REWARD, GLOBAL_EPISODES, GLOBAL_EPISODE_REWARD

SHARE_WITH_CRITIC=1
#critic network big/actor network big

GAME = 'LunarLander-v2'
LOG_DIR = './log'
MAX_EPISODES = 5000
UPDATE_EVERY = 5
GAMMA = 0.99
ENTROPY_BETA = 0.001   # not useful in this case
GLOBAL_RUNNING_REWARD = []
GLOBAL_EPISODE_REWARD = []
GLOBAL_EPISODES = 0
SESS = tf.Session()

env = gym.make(GAME)

S_ = env.observation_space.shape[0]
A_ = env.action_space.n
del env


class ACModel():
    def __init__(self, a_hidden, c_hidden, s_hidden):
        self.a_hidden=a_hidden
        self.c_hidden=c_hidden
        self.s_hidden=s_hidden
        self.s_batch=tf.placeholder(tf.float32, [None, S_], 'StateInput')
        # self.a_batch=tf.placeholder(tf.int32, [None, 1], 'Action')
        self.a_batch=tf.placeholder(tf.int32, [None,], 'Action')
        self.v_batch=tf.placeholder(tf.float32, [None, 1], 'Value')
        self.buildOps()

    def buildNetwork(self):
        weights_initializer_op=tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('shared'):
            shared_layer=tf.layers.dense(self.s_batch, self.s_hidden, tf.nn.relu, kernel_initializer=weights_initializer_op, name='shared_layer')
        with tf.variable_scope('critic'):
            critic_layer=tf.layers.dense(shared_layer, self.c_hidden, tf.nn.relu, kernel_initializer=weights_initializer_op, name='critic_layer')
            v_s=tf.layers.dense(critic_layer, 1, kernel_initializer=weights_initializer_op, name='v_s')
        with tf.variable_scope('actor'):
            actor_layer=tf.layers.dense(shared_layer, self.a_hidden, tf.nn.relu, kernel_initializer=weights_initializer_op, name='actor_layer')
            pi_s=tf.layers.dense(actor_layer, A_, tf.nn.softmax, kernel_initializer=weights_initializer_op, name='pi_s')
        return pi_s, v_s

    def buildOps(self):
        self.pi, self.v = self.buildNetwork()
        
        td = tf.subtract(self.v_batch, self.v, name='TD_error')
        self.critic_loss = tf.reduce_mean(tf.square(td))
        
        log_action_prob = tf.reduce_sum(tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0)) * tf.one_hot(self.a_batch, A_, dtype=tf.float32), axis=1, keep_dims=True)
        td_scaled_log_action_prob = log_action_prob * td
        entropy = -tf.reduce_sum(self.pi * tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0)), axis=1, keep_dims=True)  # encourage exploration
        self.exp_v = ENTROPY_BETA * entropy + td_scaled_log_action_prob
        self.actor_loss = tf.reduce_mean(-self.exp_v)

        self.shared_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared') 
        self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
        self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

        if(SHARE_WITH_CRITIC):
            self.shared_grads = tf.gradients(self.critic_loss, self.shared_params)
        else:
            self.shared_grads = tf.gradients(self.actor_loss, self.shared_params)
        
        OPT_Actor = tf.train.AdamOptimizer(name='AdamA')
        OPT_Critic = tf.train.AdamOptimizer(name='AdamC')
        self.update_actor_op = OPT_Actor.apply_gradients(zip(self.actor_grads, self.actor_params))
        self.update_critic_op = OPT_Critic.apply_gradients(zip(self.critic_grads, self.critic_params))
        self.update_shared_op = OPT_Critic.apply_gradients(zip(self.shared_grads, self.shared_params))

    def updateWeights(self, feed_dict):
        SESS.run([self.update_actor_op, self.update_critic_op, self.update_shared_op], feed_dict)
        
    def chooseAction(self, state):
        action_probs = SESS.run([self.pi], feed_dict={self.s_batch: state[np.newaxis,:]})
        action = np.random.choice(range(len(np.array(action_probs).ravel())), p=np.array(action_probs).ravel())
        return action



####################################
reward_scaling_factor=100
# use reward scaling to allow network to learn stably
env = gym.make(GAME)
AC = ACModel(100, 100, 64)
SESS.run(tf.global_variables_initializer())
steps=0
state_batch, action_batch, reward_batch = [], [], []

while GLOBAL_EPISODES < MAX_EPISODES:
    previous_state=env.reset()
    #initialise state to reset state
    episode_reward=0
    while 1:
        # env.render()
        action = AC.chooseAction(previous_state)
        current_state, reward, done, info = env.step(action)
        # reduce crashing reward to allow learning landing first
        if reward == -100: reward = -10
        steps+=1
        episode_reward+=reward

        state_batch.append(previous_state)
        action_batch.append(action)
        reward_batch.append(reward/reward_scaling_factor)
        previous_state=current_state

        if(steps%UPDATE_EVERY==0 or done):
            # value_s_t=float(not done)*SESS.run(AC.v, {AC.s_batch: current_state[np.newaxis, :]})[0,0]
            if done:
                value_s_t=0
            else:
                value_s_t=SESS.run(AC.v, {AC.s_batch: current_state[np.newaxis, :]})[0,0]

            temp_train_values=[]
            for r in reward_batch[::-1]:
                value_s_t=r+GAMMA*value_s_t
                temp_train_values.append(value_s_t)
            temp_train_values.reverse()

            state_batch, action_batch, values_batch = np.vstack(state_batch), np.array(action_batch), np.vstack(temp_train_values)
            feed_dict = {
                AC.s_batch: state_batch,
                AC.a_batch: action_batch,
                AC.v_batch: values_batch,
            }   
            AC.updateWeights(feed_dict)
            state_batch, action_batch, reward_batch = [], [], []

        if(done):
            GLOBAL_EPISODES+=1
            GLOBAL_EPISODE_REWARD.append(episode_reward)
            if(not len(GLOBAL_RUNNING_REWARD)):
                GLOBAL_RUNNING_REWARD.append(episode_reward)
            else:
                GLOBAL_RUNNING_REWARD.append(0.99 * GLOBAL_RUNNING_REWARD[-1] + 0.01 * episode_reward)
            if not env.unwrapped.lander.awake: solve = '| Landed'
            else: solve = '| ------'
            print("Ep:", GLOBAL_EPISODES, solve, "| Ep_r: %i" % GLOBAL_RUNNING_REWARD[-1],)
            break