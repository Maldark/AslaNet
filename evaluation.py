import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

model = tf.keras.models.load_model("supervised_256.h5")

# Correct is 7

# start state
state = np.zeros((1, 4), dtype=np.float32)
state = tf.convert_to_tensor(state, dtype=tf.float32)
states = tf.reshape(state, (1, 1, 4))
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

# Guess 0
state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 4/256
state2[0][1] = 1 # too low
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 140/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 117/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 117/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 72/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 32/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 32/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 32/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 32/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 20/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 20/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))


state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 4/256
state2[0][1] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 20/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 10/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 4/256
state2[0][1] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))

state2 = np.zeros((1, 4), dtype=np.float32)
state2[0][3] = 10/256
state2[0][0] = 1
state2 = tf.convert_to_tensor(state2, dtype=tf.float32)
states = tf.concat([states, tf.reshape(state2, (1, 1, 4))], axis=1)
print("State:", states)
probs = model(states).numpy()
print(probs)
print("Most likely guess", np.argmax(probs, axis=1))
