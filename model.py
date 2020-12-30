import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam


class Actor(torch.nn.Module):
    def __init__(self, num_state, num_actions, layer_1, layer_2, lr=0.0001, checkpt='ppo'):
        super(Actor, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.model = nn.Sequential(
            nn.Linear(*num_state,layer_1),
            nn.ReLU(),
            nn.Linear(layer_1,layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, num_actions),
            nn.Softmax(dim=-1)
        )

        self.optim = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        pol = self.model(state)
        return Categorical(pol)

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)


class Critic(torch.nn.Module):
    def __init__(self, num_state, layer_1, layer_2, lr=0.0001, checkpt='ppo'):
        super(Critic, self).__init__()

        self.nam_state = num_state
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.chkpt = checkpt + '_actor.ckpt'

        self.model = nn.Sequential(
            nn.Linear(*num_state,layer_1),
            nn.ReLU(),
            nn.Linear(layer_1,layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, 1)
        )

        self.optim = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        Val = self.model(state)
        return Val

    def save_model(self):
        torch.save(self.state_dict(), self.chkpt)