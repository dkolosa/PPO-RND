import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Beta, Normal
from torch.optim import Adam

import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Actor(torch.nn.Module):
    def __init__(self, num_state, num_actions, layer_1, lr=0.0001, checkpt='ppo',
                 contineous=True):
        super(Actor, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.contineous = contineous

        if self.contineous:
            self.model = nn.Sequential(
                nn.Linear(*num_state,layer_1),
                nn.ReLU(),
                nn.Linear(layer_1,layer_2),
                nn.ReLU(),
                nn.Linear(layer_2, num_actions),
                nn.Tanh()
            )
        else:
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

        if self.contineous:
            self.action_var = torch.full((num_actions,), .6 * .6).to(self.device)

    def forward(self,state):
        if self.contineous:
            mean = self.model(state)
            cov = torch.diag()
            return MultivariateNormal(mean)
        else:
            pol = self.model(state)
            return Categorical(pol)

    def save_model(self, save_dir):
        torch.save(self.state_dict(), save_dir+'/'+self.chkpt)


class Critic(torch.nn.Module):
    def __init__(self, num_state, layer_1, lr=0.0001, checkpt='ppo', contineous=False):
        super(Critic, self).__init__()

        self.nam_state = num_state
        self.layer_1 = layer_1
        self.chkpt = checkpt + '_critic.ckpt'
        self.contienous = contineous

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

    def save_model(self, save_dir):
        torch.save(self.state_dict(),save_dir+'/'+self.chkpt)


class ActorCNN(torch.nn.Module):
    def __init__(self, num_actions, layer_1, layer_2, checkpt='ppo',
                 contineous=True):
        super(ActorCNN, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.contineous = contineous

        img_size = 96*96

        if self.contineous:
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=4)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
                self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=2, stride=1)
                self.flat = Flatten()
                self.fc1 = nn.Linear(5184,layer_1)
                self.mean = nn.Linear(layer_1, num_actions)
                self.std = nn.Linear(layer_1, num_actions)


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flat(x)
        x = torch.tanh(self.fc1(x))

        mean = torch.tanh(self.mean(x))
        std_dev = F.softplus(self.std(x)) + 1.0
        return mean, std_dev


    def get_dist(self, state):
        mean, std_dev = self.forward(state)
        dist = Normal(mean, std_dev)
        return dist


class CriticCNN(torch.nn.Module):
    def __init__(self, layer_1, layer_2, checkpt='ppo'):
        super(CriticCNN, self).__init__()

        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.chkpt = checkpt + '_critic.ckpt'

        img_size= 96*96*3

        self.critic_ext = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1,layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, 1)
        )


        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
        

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        ext_val = self.critic_ext(state)
        return ext_val


class RND(torch.nn.Module):
    def __init__(self) -> None:
        super(RND, self).__init__()


        self.predictor = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512, 512)
        )


        self.target = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(4096, 512)
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.predictor.to(self.device)
        self.target.to(self.device)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                torch.nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()


    def forward(self, next_state):

        predict = self.predictor(next_state)
        target = self.target(next_state)

        return predict, target
