import os
from typing import Tuple

import tensorflow as tf
import numpy as np


class Agent():

    def __init__(self) -> None:
        pass


    def get_action(self) -> Tuple[tf.Tensor]:
        pass

    def get_transitions(self) -> Tuple[np.ndarray]:
        pass

    def train(self) -> None:
        pass

    def save(self) -> None:
        pass

    def load(self) -> None:
        pass