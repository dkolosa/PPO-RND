import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Input


class ActorCriticCNN(keras.Model):
    def __init__(self, n_actions) -> None:
        super(ActorCriticCNN, self).__init__()

        self.actor = keras.Sequential([
            Input(shape=(96,96,3)),
            # Conv2D(32, 8, activation='relu'),
            Conv2D(32, 6, activation='relu'),
            Conv2D(32, 3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            # Dense(128, activation='tanh'),
            Dense(n_actions, activation='relu')
        ])

        self.critic = keras.Sequential([
            # Conv2D(32, 8, activation='relu'),
            Conv2D(32, 6, activation='relu'),
            Conv2D(32, 3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, activation=None)
        ])

    def call(self, state):

        policy = self.actor(state)
        value = self.critic(state)
        return policy, value
        


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)

        return q
