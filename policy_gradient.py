import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import itertools

tf.enable_eager_execution()


class GuessEnv:
    def __init__(self):
        self.nA = 8
        self.number_of_rounds = 3
        self.reset()

    def reset(self, correct=None):
        self.correct = correct if correct is not None else np.random.randint(self.nA)
        self.round = 0
        return np.zeros((1, 4), dtype=np.float32)  # Start state is just zeros

    def step(self, action):
        assert 0 <= action < self.nA
        self.round += 1
        done = False
        if self.round == self.number_of_rounds:
            done = True
        state = np.zeros((1, 4), dtype=np.float32)
        state[0][3] = action / self.nA  # instead of one hot encoding
        if action == self.correct:
            state[0][2] = 1  # Correct guess!
            return state, 1., done, {"actual": self.correct}
        elif action < self.correct:
            state[0][0] = 1  # Too low
            return state, -1., done, {"actual": self.correct}
        else:
            state[0][1] = 1  # Too high
            return state, -1., done, {"actual": self.correct}


env = GuessEnv()
discount_factor = 0.95
learning_rate = 1e-2
num_episodes = 5000
epochs = 1
model_name = "supervised_{}.h5".format(env.nA)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(None, 4)),  # batch size, sequence, features
    tf.keras.layers.Dense(env.nA, activation="softmax")
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


def reward_guided_loss(model, x, y, rewards):
    y_ = model(x)
    neg_log_prob = tf.losses.sparse_softmax_cross_entropy(y, y_)
    return tf.reduce_mean(neg_log_prob * rewards)


def policy_grad(model, inputs, targets, rewards):
    with tfe.GradientTape() as tape:
        loss_value = reward_guided_loss(model, inputs, targets, rewards)
    return tape.gradient(loss_value, model.variables)


def discount_and_norm_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0
    for t in reversed(range(len(episode_rewards))):
        cumulative = cumulative * discount_factor + episode_rewards[t]
        discounted_episode_rewards[t] = cumulative
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return discounted_episode_rewards


def train():
    for epoch in range(epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        inputs = np.zeros((num_episodes, env.number_of_rounds, 4), dtype=np.float32)
        labels = np.zeros(num_episodes, dtype=np.int32)

        num_correct_guesses = 0
        for episode in range(num_episodes):
            inputs = np.zeros((num_episodes, env.number_of_rounds, 4), dtype=np.float32)
            labels = np.zeros(num_episodes, dtype=np.int32)
            state = env.reset(correct=episode % env.nA)
            states = tf.zeros((1, 0, 4), dtype=np.float32)
            episode_rewards = []
            guessed_right = False

            for t in itertools.count():
                states = tf.concat([states, tf.reshape(state, (1, 1, 4))], axis=1)
                action_probs = model(states)
                action = np.random.choice(np.arange(env.nA), p=action_probs.numpy()[0])

                # Take action
                next_state, reward, done, debug = env.step(action)
                print("Guessing", action, "actual", env.correct)

                episode_rewards.append(reward)
                state = next_state
                if not guessed_right and reward == 1:
                    guessed_right = True
                    num_correct_guesses += 1

                if done:
                    break

            inputs[episode] = states
            labels[episode] = env.correct

            # Policy Gradient update
            discounted_reward = discount_and_norm_rewards(episode_rewards)
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            input = inputs[None, episode]
            label = labels[None, episode]
            grads = policy_grad(model, input, label, discounted_reward)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step=tf.train.get_or_create_global_step())
            epoch_loss_avg(reward_guided_loss(model, inputs, labels, discounted_reward))  # add current batch loss
            epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), labels)
            print("REINFORCE loss", reward_guided_loss(model, input, label, discounted_reward))
            print(guessed_right)


        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # grads = softmax_grad(model, inputs, labels)
        #
        # grads = [tf.clip_by_value(g, -1., 1.) for g in grads]  # apply gradient clipping
        # optimizer.apply_gradients(zip(grads, model.variables),
        #                           global_step=tf.train.get_or_create_global_step())
        #
        # epoch_loss_avg(softmax_loss(model, inputs, labels))  # add current batch loss
        # epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), labels)

        # # Policy Gradient update
        # rewards = (rewards - rewards.mean()) / rewards.std()
        # reinforce_episode = np.random.randint(num_episodes)
        # input = inputs[None, reinforce_episode]
        # label = labels[None, reinforce_episode]
        # print("rewards", rewards)
        # grads = policy_grad(model, input, label, rewards)
        # optimizer.apply_gradients(zip(grads, model.variables),
        #                           global_step=tf.train.get_or_create_global_step())
        # epoch_loss_avg(reward_guided_loss(model, inputs, labels, rewards))  # add current batch loss
        # epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), labels)
        # print("REINFORCE loss", reward_guided_loss(model, input, label, rewards))


        if epoch % 20 == 0:
            model.save(model_name, include_optimizer=False)
            print("Model saved!")


train()

model.save(model_name, include_optimizer=False)


