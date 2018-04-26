import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import itertools

tf.enable_eager_execution()


class GuessEnv:
    def __init__(self):
        self.nA = 64
        self.number_of_rounds = 8
        self.reset()

    def reset(self, correct=None):
        self.correct = correct if correct is not None else np.random.randint(self.nA)
        state, _, _, _ = self.step(0)  # Initial state is a guess of zero.
        self.round = 0
        return state

    def step(self, action):
        assert 0 <= action < self.nA
        self.round += 1
        if self.round == self.number_of_rounds:
            done = True
        else:
            done = False
        state = np.zeros((1, 4), dtype=np.float32)
        state[0][3] = action / self.nA  # instead of one hot encoding
        if action == self.correct:
            state[0][2] = 1  # Correct guess!
            return tf.convert_to_tensor(state, dtype=tf.float32), 1, done, {"actual": self.correct}
        elif self.correct < action:
            state[0][0] = 1  # Go lower!
            return tf.convert_to_tensor(state, dtype=tf.float32), -1, done, {"actual": self.correct}
        else:
            state[0][1] = 1  # Go higher!
            return tf.convert_to_tensor(state, dtype=tf.float32), -1, done, {"actual": self.correct}


env = GuessEnv()

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(None, 4)),  # batch size, sequence, features
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
num_episodes = env.nA * 2
epochs = 1000
# batch_size = 512
# discount_factor = 0.95
# epsilon = 0.1

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for epoch in range(epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    inputs = np.zeros((num_episodes, env.number_of_rounds, 4), dtype=np.float32)
    labels = np.zeros(num_episodes, dtype=np.int32)

    episode_rewards = 0
    for episode in range(num_episodes):
        state = env.reset(correct=episode % env.nA)

        states = tf.reshape(state, (1, 1, 4))
        for t in itertools.count():
            action_probs = model(states)
            action = np.random.choice(np.arange(env.nA), p=action_probs.numpy()[0])

            # Take action
            next_state, reward, done, debug = env.step(action)
            next_states = tf.concat([states, tf.reshape(next_state, (1, 1, 4))], axis=1)

            episode_rewards += reward
            states = next_states

            if done:
                break

        inputs[episode] = states
        labels[episode] = env.correct

    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    grads = grad(model, inputs, labels)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step=tf.train.get_or_create_global_step())

    epoch_loss_avg(loss(model, inputs, labels))  # add current batch loss
    epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), labels)

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Avg. reward: {:.3f}".format(epoch, epoch_loss_avg.result(),
                                                                                     epoch_accuracy.result(),
                                                                                     episode_rewards / num_episodes))

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
