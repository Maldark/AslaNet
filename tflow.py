import itertools
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()


class GuessEnv:
    def __init__(self):
        self.nA = 64
        self.reset()

    def reset(self):
        self.correct = np.random.randint(2)
        # self.correct = 5
        self.round = -1
        state, _, _, _ = self.step(self.nA // 2)
        return state

    def step(self, action):
        assert 0 <= action < self.nA
        self.round += 1
        done = False
        if self.round == 10:
            done = True
        # (low, high, correct, guess)
        # state = tf.convert_to_tensor(np.random.random((1,4)), dtype=tf.float32)

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
    tf.keras.layers.LSTM(16, input_shape=(None, 4), return_sequences=True),  # batch size, sequence, features
    tf.keras.layers.TimeDistributed(env.nA, activation='softmax')
    # tf.keras.layers.Dropout(0.5)
])
print(model.summary())


def loss(model, x, y):
    y_ = model(x)
    return tf.reduce_sum(tf.square(y - y_))


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
    inputs = tf.zeros((1, 0, 4))
    actions = 0
    actionsTotal = 0
    for t in itertools.count():
        inputs = tf.concat([inputs, tf.reshape(state, (1, 1, 4))], axis=1)

        # Choose action
        q = model(inputs)
        action = tf.argmax(q, axis=1).numpy()[0]
        if np.random.rand(1) < e:
            action = np.random.randint(env.nA)

        # Take action
        next_state, reward, done, debug = env.step(action)

        next_inputs = tf.concat([inputs, tf.reshape(next_state, (1, 1, 4))], axis=1)
        total_reward += reward

        next_q = model(next_inputs)
        target_entry = reward + discount_factor * np.max(next_q)
        target = q.numpy()
        target[0][action] = target_entry

        # Optimize the model
        grads = grad(model, inputs, target)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, inputs, target))
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), target)


        # print("Guessed", action, "got reward", reward, "correct", debug["actual"])
        if done:
            e = 1./((episode/50) + 10)
            break

        actions += action
        actionsTotal += action
        # if t % 5 == 0 and t > 0:
        #     print("For t", t, "Avg guess last 20", actions/5, "Avg guess overall", actionsTotal/t, "where actual", debug["actual"])
        #     actions = 0

        state = next_state
    print("Total reward", total_reward)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    print("Episode {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(episode,
                                                                  epoch_loss_avg.result(),
                                                                  epoch_accuracy.result()))
