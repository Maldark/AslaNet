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
        self.round = 0
        return np.zeros((1, 4), dtype=np.float32)  # Start state is just zeros

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
    tf.keras.layers.LSTM(16, input_shape=(None, 4)),  # batch size, sequence, features
    tf.keras.layers.Dense(env.nA, activation="softmax")
])
print(model.summary())


def loss(model, observation, targetQ):
    currentQ = model(observation)
    return tf.reduce_sum(tf.square(targetQ, currentQ))


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


learning_rate = 1e-2
num_episodes = env.nA * 2
epochs = 1000
# batch_size = 512
# discount_factor = 0.95
epsilon = 0.1

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

for epoch in range(epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    inputs = np.zeros((num_episodes, env.number_of_rounds, 4), dtype=np.float32)
    labels = np.zeros(num_episodes, dtype=np.int32)

    episode_rewards = 0
    num_correct_guesses = 0
    rounds_before_correct = 0
    for episode in range(num_episodes):
        state = env.reset(correct=episode % env.nA)
        states = tf.zeros((1, 0, 4), dtype=np.float32)

        for t in itertools.count():
            states = tf.concat([states, tf.reshape(state, (1, 1, 4))], axis=1)
            q = model(states)
            if np.random.rand(1) < epsilon:
                action = np.random.choice(np.arange(env.nA))
            else:
                action = tf.argmax(q, 1)

            # Take action
            next_state, reward, done, debug = env.step(action)
            next_q = model(next_state)
            q[]

            # log statistics
            if reward == 1:
                num_correct_guesses += 1
                rounds_before_correct += t

            episode_rewards += reward
            state = next_state

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

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Avg. reward: {:.3f}, Correct episodes: {}/{} with avg. rounds: {:.3f}"
          .format(epoch, epoch_loss_avg.result(),
                  epoch_accuracy.result(),
                  episode_rewards / num_episodes,
                  num_correct_guesses,
                  num_episodes,
                  rounds_before_correct / num_correct_guesses))

    if epoch % 50 == 0:
        model.save("qlearning.h5")
        print("Model saved!")


model.save("qlearning.h5")

