import os
import itertools
import random
from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

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

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(None, 4)),  # batch size, sequence, features
    tf.keras.layers.Dense(env.nA, activation="softmax")
])
print(model.summary())

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


learning_rate = 1e-2
num_episodes = 30
epochs = 20
# batch_size = 512
# discount_factor = 0.95
# epsilon = 0.1

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

for epoch in range(epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    results = []
    replay_memory = []

    episode_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()

        guesses = []
        # Training loop - using batches of 32
        states = tf.reshape(state, (1, 1, 4))
        for t in itertools.count():
            # Choose action epsilon greedy
            # if np.random.rand(1) <= epsilon:
            #     action = np.random.randint(env.nA)
            # else:
            #     q = model(states)
            #     action = tf.argmax(q, axis=1).numpy()[0]
            action_probs = model(states)
            action = np.random.choice(np.arange(env.nA), p=action_probs.numpy()[0])

            # Take action
            next_state, reward, done, debug = env.step(action)
            next_states = tf.concat([states, tf.reshape(next_state, (1, 1, 4))], axis=1)
            replay_memory.append(Transition(states, action, reward, next_states, done))
            guesses.append(action)


            episode_rewards += reward
            states = next_states
            if done:
                break

        results.append((states, env.correct))
        # print(guesses, env.correct)

    l = 0
    for x, y in results:
        y = tf.constant([y])
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Avg. reward: {:.3f}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result(), episode_rewards/num_episodes))

model.save("supervised.h5")

    # samples = random.sample(replay_memory, batch_size)
    # samples = replay_memory
    # states_batch, action_batch, reward_batch, next_states_batch, done_batch = zip(*samples)
    # targets = model(states_batch).numpy()
    # q_values_next = model(next_states_batch)
    # targets[:, action_batch] = reward_batch + discount_factor * np.max(q_values_next, axis=1)*np.invert(done_batch)
    #
    # grads = grad(model, states_batch, targets)
    # optimizer.apply_gradients(zip(grads, model.variables),
    #                           global_step=tf.train.get_or_create_global_step())
    #
    # l = loss(model, states_batch, targets)
