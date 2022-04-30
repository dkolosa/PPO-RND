from typing import Tuple
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, ReLU, Softmax

class ActorCritic(keras.Model):
    def __init__(self, num_inputs, num_actions) -> None:
        super(ActorCritic).__init__()
        num_actions
        layer_1 = 512
        layer_2 = 512

        self.actor = keras.Sequential(
            Dense(layer_1),
            ReLU(),
            Dense(layer_2),
            ReLU(),
            Dense(num_actions),
            Softmax()
        )

        self.critic = keras.Sequential(
            Dense(layer_1),
            ReLU(),
            Dense(layer_2),
            ReLU(),
            Dense(1)
        )

    def call(self, state) -> Tuple[tf.Tensor, tf.Tensor]:
        pol = self.actor(state)
        value = self.critic(state)
        return pol, value