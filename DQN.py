from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical
import random
import numpy as np
import pandas as pd
from operator import add


class DQNAgent(object):

    def __init__(self, action_space, input_dim, mode='training', load_weights=False):
        self.mode = mode
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.action_space = action_space
        self.model = self.network(input_dim=input_dim, load_weights=load_weights)
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def network(self, input_dim, load_weights=False):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=self.action_space.n, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if load_weights:
            model.load_weights('weights.hdf5')
        return model

    def remember(self, state, action, reward, next_state, done):
        # if reward != 0:
        self.memory.append((state, action, reward, next_state, done))

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        """ train short memory base on the new action and state"""
        self.train(state, action, reward, next_state, done)
        # store the new data into a long term memory
        self.remember(state, action, reward, next_state, done)

    def dream(self):
        batch_size = 1000
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for args in minibatch:
            self.train(*args)


    def predict(self, state, game_counter):
        act = None
        if self.mode == 'training':
            #self.epsilon is set to give randomness to actions
            self.epsilon = 80 - game_counter

            #perform random actions based on self.epsilon, or choose the action
            if random.randint(0, 200) < self.epsilon:
                act = self.action_space.sample()
            else:
                # predict action based on the old state
                prediction = self.model.predict(state)[0]
                act = np.argmax(prediction)
        elif self.mode == 'testing':
            # predict action based on the old state
            prediction = self.model.predict(state)[0]
            act = np.argmax(prediction)
        elif self.mode == 'random':
            act = self.action_space.sample()
        else:
            raise Exception(f'Invalid mode {mode}')
        return act

    def save_model(self):
        self.model.save_weights('weights.hdf5')
