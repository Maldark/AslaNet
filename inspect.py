import os
import itertools
import random
from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

class GuessEnv:
    def __init__(self):
        self.nA = 64
        self.max_rounds = 8
        self.reset()

    def reset(self):
        # self.correct = np.random.randint(self.nA)
        self.correct = np.random.randint(4)
        self.round = 0
        state, _, _, _ = self.step(np.random.randint(self.nA))
        return state

    def step(self, action):
        assert 0 <= action < self.nA
        self.round += 1
        if self.round == self.max_rounds:
            done = True
        else:
            done = False
        state = np.zeros((1, 4), dtype=np.float32)
        state[0][3] = action
        if action == self.correct:
            state[0][2] = 1
            return tf.convert_to_tensor(state, dtype=tf.float32), 1, True, {"actual": self.correct}
        elif action < self.correct:
            state[0][0] = 1
            return tf.convert_to_tensor(state, dtype=tf.float32), -1, done, {"actual": self.correct}
        else:
            state[0][1] = 1
            return tf.convert_to_tensor(state, dtype=tf.float32), -1, done, {"actual": self.correct}


env = GuessEnv()

model = tf.keras.models.load_model("supervised.h5")
state = np.zeros((1, 4), dtype=np.float32)
state[0][3] = 3
state[0][1] = 1

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 1
state2[0][1] = 1
statetf1 = tf.convert_to_tensor(state, dtype=tf.float32)
statetf2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.reshape(statetf1, (1, 1, 4))
probs = model(states).numpy()
print(probs)
print(np.argmax(probs, axis=1))
states = tf.concat([states, tf.reshape(statetf2, (1, 1, 4))], axis=1)
print("")
print(states)
probs = model(states).numpy()
print(probs)
print(np.argmax(probs, axis=1))

