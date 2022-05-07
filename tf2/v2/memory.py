from os import stat
import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.states_ = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
    
    def get_mems(self):
        return np.array(self.states),\
            np.array(self.states_), \
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        return [indices[i:i+self.batch_size] for i in batch_start]



    def store_memory(self, state, state_, action, probs, vals, reward, done):
        self.states.append(state)
        self.states_.append(state_)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.states_ = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
