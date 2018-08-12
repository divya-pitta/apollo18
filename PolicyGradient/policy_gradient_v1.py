"""
Policy Gradient in Keras
"""
import gym
import os
import glob
import time
import pickle
import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers


class Agent(object):

    def __init__(self, n_input, n_output, n_hidden=[32, 32],
                 reward_decay=0.99, learning_rate=0.001):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.gamma = reward_decay
        self.lr = learning_rate

        self.__build_model()
        self.__build_train_fn()

    def __build_model(self):
        self.X = layers.Input(shape=(self.n_input,))
        net = self.X

        for h_dim in self.n_hidden:
            net = layers.Dense(h_dim, activation="relu", use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros')(net)

        net = layers.Dense(self.n_output, activation="softmax", use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros')(net)
        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        action_prob = self.model.output
        action_onehot = K.placeholder(shape=(None, self.n_output), name="action_onehot")
        discount_reward = K.placeholder(shape=(None,), name="discount_reward")

        log_action_prob = K.log(action_prob)
        log_action_prob = K.sum(action_onehot * log_action_prob, axis=1)
        loss = - log_action_prob * discount_reward
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot,
                                           discount_reward],
                                   outputs=[loss],
                                   updates=updates)

    def get_action(self, state):
        shape = state.shape
        if len(shape) == 1:
            state = np.expand_dims(state, axis=0)
        action_prob = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(self.n_output), p=action_prob)

    def fit(self, states, actions, rewards):
        action_onehot = np_utils.to_categorical(actions, num_classes=self.n_output)
        discount_reward = self.compute_discounted_R(rewards)
        return self.train_fn([states, action_onehot, discount_reward])[0]

    def compute_discounted_R(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        discounted_r -= discounted_r.mean() / discounted_r.std()
        return discounted_r

    def save(self, save_path):
        self.model.save_weights(save_path)

    def load(self, load_path):
        self.model.load_weights(load_path)


def run_episode(env, agent, render):
    done = False
    states, actions, rewards = [], [], []

    state = env.reset()
    total_reward = 0

    while not done:
        if render: env.render()
        action = agent.get_action(state)
        state_, reward, done, info = env.step(action)
        total_reward += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = state_

        if done:
            loss = agent.fit(np.array(states), np.array(actions), np.array(rewards))
            if render: env.close()

    return total_reward, loss

def main():
    try:
        version = 1
        env = gym.make("LunarLander-v2")
        n_input = env.observation_space.shape[0]
        n_output = env.action_space.n
        agent = Agent(n_input, n_output, [32, 32])
        folder = "weights/LunarLander-v2_v{}".format(version)
        pkl_file = "pg_v{}.pkl".format(version)

        os.makedirs(folder, exist_ok=True)
        list_of_files = glob.glob(folder + "/*")
        if len(list_of_files) > 0:
            latest_file = max(list_of_files, key=os.path.getctime)
            latest_file = 'weights/LunarLander-v2_v1/100.h5'
            agent.load(latest_file)

        run_episode(env, agent, True)
        env.close()
        return


        rewards, losses, times = list(), list(), list()
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                rewards, losses, times = pickle.load(f)

        for episode in range(len(rewards), 100000):
            tic = time.clock()
            render = False
            if episode % 100 == 0:
                # render = True
                save_path = folder + "/{}.h5"
                agent.save(save_path.format(episode))
                with open(pkl_file, 'wb') as f:
                    pickle.dump([rewards, losses, times], f)
            toc = time.clock()
            reward, loss = run_episode(env, agent, render)
            rewards.append(reward)
            losses.append(loss)
            times.append(toc - tic)
            print("====================")
            print("Episode: ", episode)
            print("Loss: ", loss)
            print("Running Loss: ", sum(losses) / len(losses))
            print("Reward: ", reward)
            print("Running Reward: ", sum(rewards) / len(rewards))
            print("Time: ", toc - tic)
        if not env.unwrapped.lander.awake:
            print("+++++++LANDED++++++++")
    finally:
        env.close()

if __name__ == '__main__':
    main()
