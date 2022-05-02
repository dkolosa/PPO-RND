from typing import Tuple
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense

class ActorCritic(keras.Model):
    def __init__(self, num_inputs, num_actions) -> None:
        super(ActorCritic, self).__init__()
        num_actions
        layer_1 = 128
        layer_2 = 128

        self.actor = keras.Sequential([
            Dense(layer_1, activation='relu'),
            Dense(layer_2, activation='relu'),
            Dense(num_actions, activation='softmax')
        ])

        self.critic = keras.Sequential([
            Dense(layer_1, activation='relu'),
            Dense(layer_2, activation='relu'),
            Dense(num_actions, activation=None)
        ])

    def call(self, state) -> Tuple[tf.Tensor, tf.Tensor]:
        pol = self.actor(state)
        value = self.critic(state)
        return pol, value