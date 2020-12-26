import numpy as np
import torch
from model import Actor, Critic
import torch.optim as optim


class Memory()
    def __init__(self,batch):
        # init state, action, reward, state_, done

        self.batch = batch

    def get_memory(self):
        pass

    def store_memory(self,state, action, reward, state_, done)

        pass

    def clear_memory(self):

        pass

class Agent():
    def __init__(self, ep=0.1, beta=3, c1=0.1, layer_1_nodes=512, layer_2_nodes=256):
        
        self.ep = ep
        self.beta = beta

        self.actor = Actor(state, layer_1_nodes, layer_2_nodes)
        self.critic = Critic(state, layer_1_nodes, layer_2_nodes)
        

    def train(self, state):

        #run policy
        # Compute avg estimates A_n
        # A(V(s))

        At = self.critic(s)

        # L = L^clip - c1L^VF

        # L^c = np.clip(r*At,1-ep, 1+ep)*At 
        pass
        


