import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import itertools

tf.enable_eager_execution()


class GuessEnv:
    def __init__(self):
        self.number_of_actions = 256
        self.number_of_rounds = 8 # one should optimally only spend log(nA) rounds
        self.reset()

    def reset(self, correct=None):
        self.correct = correct if correct is not None else np.random.randint(self.number_of_actions)
        self.round = 0
        return np.zeros((1, 4), dtype=np.float32)  # Start state is just zeros

    def step(self, action):
        assert 0 <= action < self.number_of_actions
        self.round += 1
        done = False
        if self.round == self.number_of_rounds:
            done = True
        state = np.zeros((1, 4), dtype=np.float32)
        state[0][3] = action / self.number_of_actions  # stabilize network by having all input numbers between 0-1.
        if action == self.correct:
            state[0][2] = 1  # Correct guess!
            return state, 1, done, {"actual": self.correct}
        elif action < self.correct:
            state[0][0] = 1  # Too low
            return state, -1, done, {"actual": self.correct}
        else:
            state[0][1] = 1  # Too high
            return state, -1, done, {"actual": self.correct}


env = GuessEnv()
learning_rate = 1e-3
num_episodes = env.number_of_actions * 2
epochs = 2000
model_name = "supervised_{}.h5".format(env.number_of_actions)

# Use LSTMs to model "memory", the model estimates the next guess as a combination of previous and current results.
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(None, 4)),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(env.number_of_actions, activation="softmax")
])
print(model.summary())

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


def softmax_loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(y, y_)


def softmax_grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = softmax_loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def train():
    for epoch in range(epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        inputs = np.zeros((num_episodes, env.number_of_rounds, 4), dtype=np.float32)
        labels = np.zeros(num_episodes, dtype=np.int32)

        num_correct_guesses = 0
        for episode in range(num_episodes):
            # We fix the correct value to stabilize the learning process
            state = env.reset(correct=episode % env.number_of_actions)
            states = tf.zeros((1, 0, 4), dtype=np.float32)
            episode_rewards = 0
            guessed_right = False

            for _ in itertools.count():
                states = tf.concat([states, tf.reshape(state, (1, 1, 4))], axis=1)
                action_probs = model(states)
                action = np.random.choice(np.arange(env.number_of_actions), p=action_probs.numpy()[0])

                # Take action
                next_state, reward, done, debug = env.step(action)
                state = next_state

                # Log statistics
                episode_rewards += reward
                if not guessed_right and reward == 1:
                    guessed_right = True
                    num_correct_guesses += 1

                if done:
                    break

            inputs[episode] = states
            labels[episode] = env.correct

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        grads = softmax_grad(model, inputs, labels)

        # Apply gradient clipping to avoid exploding gradients by cutting off gradients to values between -1 and 1.
        grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        epoch_loss_avg(softmax_loss(model, inputs, labels))  # add current batch loss
        epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), labels)

        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Avg. reward: {:.3f}, Correct episodes: {}/{}"
              .format(epoch, epoch_loss_avg.result(),
                      epoch_accuracy.result(),
                      episode_rewards / num_episodes,
                      num_correct_guesses,
                      num_episodes))
        with open("result.csv", "a") as f:
            f.write("{:03d}, {:.3f}, {:.3%}, {:.3f}, {}/{}"
                    .format(epoch, epoch_loss_avg.result(),
                    epoch_accuracy.result(),
                    episode_rewards / num_episodes,
                    num_correct_guesses,
                    num_episodes))

        if epoch % 20 == 0:
            model.save(model_name, include_optimizer=False)
            print("Model saved!")


train()

model.save(model_name, include_optimizer=False)
