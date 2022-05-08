import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import Beta, Normal
from torch.optim import Adam


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Actor(torch.nn.Module):
    def __init__(self, num_state, num_actions, layer_1, layer_2, lr=0.0001, checkpt='ppo',
                 contineous=False):
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
    def __init__(self, num_state, layer_1, layer_2, lr=0.0001, checkpt='ppo', contineous=False):
        super(Critic, self).__init__()

        self.nam_state = num_state
        self.layer_1 = layer_1
        self.layer_2 = layer_2
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
    def __init__(self, num_state, num_actions, layer_1, layer_2, lr=0.0001, checkpt='ppo',
                 contineous=True):
        super(ActorCNN, self).__init__()

        self.chkpt = checkpt + '_actor.ckpt'

        self.contineous = contineous

        img_size = 96*96

        if self.contineous:
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1)
                self.flat = Flatten()
                self.fc1 = nn.Linear(3136,layer_1)
                self.fc2 = nn.Linear(layer_1,layer_2)
                self.mean = nn.Linear(layer_2, num_actions)
                self.std = nn.Linear(layer_2, num_actions)

        else:
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1)
                self.flat = Flatten()
                self.fc1 = nn.Linear(3136,layer_1)
                self.fc2 = nn.Linear(layer_1,layer_2)
                self.mean = nn.Softmax(dim=-1)

        def calc_cnnweights():
            input = torch.zeros((1, 3, 86, 86))
            model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1),
                nn.ReLU())

            x = model(input)
            return x.shape[1]

        self.optim = Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,state):
        if self.contineous:
            x = torch.relu(self.conv1(state))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.flat(x)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))

            mean = self.mean(x)
            std_dev = self.std(x)
            dist = Normal(mean, std_dev.exp())
            return dist
            
        else:
            pol = self.model(state)
            return Categorical(pol)



class CriticCNN(torch.nn.Module):
    def __init__(self, num_state, layer_1, layer_2, lr=0.0001, checkpt='ppo', contineous=False):
        super(CriticCNN, self).__init__()

        self.nam_state = num_state
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.chkpt = checkpt + '_critic.ckpt'
        self.contienous = contineous

        img_size= 96*96*3

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(2304, layer_1),
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