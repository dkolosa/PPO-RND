from typing import Tuple
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

class ActorCritic(keras.Model):
    def __init__(self) -> None:
        super(ActorCritic).__init__()

        pass

    def call(self) -> Tuple[tf.Tensor, tf.Tensor]:
        pass