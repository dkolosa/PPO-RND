import torch
import torch.nn as nn

class Actor(torch.nn.Module):
    def __init__(self, num_state, num_actions, layer_1, layer_2, lr, checkpt):
        super(Actor, self).__init__()

        self.nam_state = num_state
        self.num_actions = num_actions
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.chkpt = checkpt + '_actor.ckpt'

        self.model = nn.Sequential(
            nn.Linear(num_state,layer_1),
            nn.ReLU()
            nn.Linear(num_state,layer_1),
            nn.ReLU()
            nn.Linear(layer_2, num_actions)
            nn.Softmax(ndim=-1)
        )

    def forward(self,state):
        pol = self.model(state)


class Critic(torch.nn.Module):
    def __init__(self, num_actions, layer_1, layer_2, lr, checkpt):
        super(Actor, self).__init__()

        self.nam_state = num_state
        self.num_actions = num_actions
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.chkpt = checkpt + '_actor.ckpt'

        self.model = nn.Sequential(
            nn.Linear(num_state,layer_1),
            nn.ReLU()
            nn.Linear(num_state,layer_1),
            nn.ReLU()
            nn.Linear(layer_2, 1)
        )


    def forward(self,state):
        Val = self.model(state)