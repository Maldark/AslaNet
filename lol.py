import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

model = tf.keras.models.load_model("supervised.h5")

# Correct is 2, guess 3, which is higher
state = np.zeros((1, 4), dtype=np.float32)
state[0][3] = 5/6
state[0][1] = 1

# Afterwards, guess 1, lower
state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 2/6
state2[0][0] = 1
statetf1 = tf.convert_to_tensor(state, dtype=tf.float32)
statetf2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.reshape(statetf1, (1, 1, 4))
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

states = tf.concat([states, tf.reshape(statetf2, (1, 1, 4))], axis=1)
print("")
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess:", np.argmax(probs, axis=1))

