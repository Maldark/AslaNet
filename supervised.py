import tensorflow as tf
import numpy as np
import itertools


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
learning_rate = 1e-2
num_episodes = env.number_of_actions * 2
epochs = 2000
model_name = "supervised_{}.h5".format(env.number_of_actions)

# Use LSTMs to model "memory", the model estimates the next guess as a combination of previous and current results.
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(None, 4)),
    tf.keras.layers.Dense(env.number_of_actions, activation="softmax")
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())


def train():
    for epoch in range(epochs):
        inputs = np.zeros((num_episodes, env.number_of_rounds, 4), dtype=np.float32)
        labels = np.zeros((num_episodes, env.number_of_actions), dtype=np.int32)

        num_correct_guesses = 0
        for episode in range(num_episodes):
            # We fix the correct value to stabilize the learning process
            state = env.reset(correct=episode % env.number_of_actions)
            states = np.zeros((1, 0, 4), dtype=np.float32)
            episode_rewards = 0
            guessed_right = False

            for _ in itertools.count():
                states = np.concatenate([states, np.reshape(state, (1, 1, 4))], axis=1)
                action_probs = model.predict(states)
                action = np.random.choice(np.arange(env.number_of_actions), p=action_probs[0])

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
            labels[episode] = np.eye(env.number_of_actions)[env.correct]

        model.fit(inputs, labels)
        model.save(model_name)
        print("Epoch {}/{}, Avg. reward: {:.3f}, Correct episodes: {}/{}".format(epoch, epochs, episode_rewards / num_episodes, num_correct_guesses, num_episodes))


train()
