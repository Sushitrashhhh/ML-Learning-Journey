import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import time

# Example environment space (CartPole uses vector states, so no Conv2D)
OBSERVATION_SPACE_VALUES = (4,)  # CartPole: 4 numbers
ACTION_SPACE_SIZE = 2            # CartPole: left or right

REPLAY_MEMORY_SIZE = 50_000
MODEL_NAME = "DQN"

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        # For vector input (CartPole, MountainCar)
        model.add(Dense(64, input_shape=OBSERVATION_SPACE_VALUES, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(ACTION_SPACE_SIZE, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]


# Modern TensorBoard wrapper
class ModifiedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):  # override
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()
