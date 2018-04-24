import itertools
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

class GuessEnv:
    def __init__(self):
        self.nA = 16
        self.reset()

    def reset(self):
        self.correct = np.random.randint(self.nA)
        self.round = -1
        state, _, _, _ = self.step(self.nA // 2)
        return state

    def step(self, action):
        assert 0 <= action < self.nA
        self.round += 1
        done = False
        # if self.round == 8:
        #     done = True
        # (low, high, correct, guess)
        # state = tf.convert_to_tensor(np.random.random((1,4)), dtype=tf.float32)

        state = np.zeros((1, 4), dtype=np.float32)
        state[0][3] = action / float(self.nA)
        if action == self.correct:
            state[0][2] = 1
            return tf.convert_to_tensor(state, dtype=tf.float32), 1, True, {}
        elif action < self.correct:
            state[0][0] = 1
            return tf.convert_to_tensor(state, dtype=tf.float32), -1, False, {}
        else:
            state[0][1] = 1
            return tf.convert_to_tensor(state, dtype=tf.float32), -1, False, {}


env = GuessEnv()


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(None, None, 4)),  # input shape required
    tf.keras.layers.Dense(env.nA)
])


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.mean_squared_error(labels=y, predictions=y_)


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train_loss_results = []
train_accuracy_results = []

num_episodes = 200
discount_factor = 0.95
e = 0.1

for episode in range(num_episodes):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    state = env.reset()

    # Training loop - using batches of 32
    total_reward = 0
    for t in itertools.count():
        # Choose action (maybe epsilon greedy?)
        q = model(state)
        action = tf.argmax(q, axis=1).numpy()[0]
        if np.random.rand(1) < e:
            action = np.random.randint(env.nA)

        # Take action
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        next_q = model(next_state)
        target_entry = reward + discount_factor * np.max(next_q)
        target = q.numpy()
        target[0][action] = target_entry

        # Optimize the model
        grads = grad(model, state, target)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, state, target))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(state), axis=1, output_type=tf.int32), target)
        if done:
            break
    print(total_reward)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    print("Episode {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(episode,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
