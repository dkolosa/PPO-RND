import numpy as np
import torch
from model import Actor, Critic, ActorCNN, CriticCNN


class Memory():
    def __init__(self,batch_size):
        # init state, action, reward, state_, done
        self.state = []
        self.state_1 = []
        self.prob = []
        self.action = []
        self.reward = []
        self.done = []
        self.batch_size = batch_size

    def get_memory(self):

        return np.array(self.state),\
            np.array(self.state_1), \
            np.array(self.action),\
            np.array(self.prob),\
            np.array(self.reward),\
            np.array(self.done)
        
    def get_batches(self):
        n_states = len(self.state)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        return [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batches)]
        
    def store_memory(self, state, s_1, action, prob, reward, done):
        self.state.append(state)
        self.state_1.append(s_1)
        self.action.append(action)
        self.reward.append(reward)
        self.prob.append(prob)
        self.done.append(done)

    def clear_memory(self):
        self.state.clear()
        self.state_1.clear()
        self.action.clear()
        self.reward.clear()
        self.prob.clear()
        self.done.clear()
        

class Agent():
    def __init__(self, num_state, num_action, ep=0.2, beta=3, c1=0.1, layer_1_nodes=512, layer_2_nodes=256, batch_size=64,save_dir='models'):
        
        self.ep = ep
        self.beta = beta
        self.c1 = c1
        self.gamma = .99
        self.g_lambda = 0.95

        # self.actor = Actor(num_state, num_action, layer_1_nodes, layer_2_nodes, contineous=True)
        # self.critic = Critic(num_state, layer_1_nodes, layer_2_nodes, contineous=True)

        self.actor = ActorCNN(num_state, num_action, layer_1_nodes, layer_2_nodes,lr=0.0001, checkpt='ppo', contineous=True)
        self.critic = CriticCNN(num_state, layer_1_nodes, layer_2_nodes,contineous=True)

        self.memory = Memory(batch_size)

        self.save_dir = save_dir

    def take_action(self,state):
        with torch.no_grad():

            state = torch.tensor([state], dtype=torch.float, device=self.actor.device)
            # torch.permute(state, (2,0,1))
            print(state.shape)
            prob_dist = self.actor(state)
            action = prob_dist.sample()
            
            # action = tytorch.clamp(action, -1, 1)
            prob = torch.squeeze(prob_dist.log_prob(action)).cpu().detach().numpy()
            action = torch.squeeze(action).cpu().detach().numpy()


        return prob, action

    def preprocess_image(self,image):
        # pytorch image: C x H x W
        # image = image[0:86, 0:86, 0:3]
        image_swp = np.swapaxes(image, -1, 0)
        image_swp = np.swapaxes(image_swp,-1, -2)
        return image_swp/255.0

    def store_memory(self, state, s_1, action, prob, reward, done):
        self.memory.store_memory(state, s_1, action, prob, reward, done)

    def calculate_adv_ret(self, state, state_1, reward, done):
        with torch.no_grad():
            value = self.critic(state)
            value_1 = self.critic(state_1)

            delta = reward + self.gamma * value_1 - value
            delta = delta.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(delta[::-1], done[::-1]):
                advantage = dlt + self.gamma * self.g_lambda * adv[-1] * \
                            (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = torch.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            returns = adv + value
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns


    def train(self):
        epochs = 5
            
        state_mem, state_1_mem, action_mem, prob_mem, reward_mem, done_mem = self.memory.get_memory()
        states = torch.tensor(state_mem, dtype=torch.float, device=self.actor.device)
        states_1 = torch.tensor(state_1_mem, dtype=torch.float, device=self.actor.device)
        actions = torch.tensor(action_mem, dtype=torch.float, device=self.actor.device)
        old_probs = torch.tensor(prob_mem, dtype=torch.float, device=self.actor.device)
        rewards = torch.tensor(reward_mem, dtype=torch.float, device=self.actor.device)
        advantage, returns = self.calculate_adv_ret(state, states_1, rewards, done_mem)

        for epoch in range(epochs):
            batches = self.memory.get_batches()

            for batch in batches:

                # calculate r_t(theta)
                state = states[batch]
                old_prob = old_probs[batch]
                action = actions[batch]
                breakpoint()

                dist_new = self.actor(states)
                entropy = dist_new.entropy().sum(1, keepdims=True)
                prob_new = dist_new.log_prob(action)

                r_t = (prob_new.sum(1, keepdims=1) - old_prob.sum(1, keepdims=1)).exp()        

                # L_clip
                prob_clip = torch.clamp(r_t, 1-self.ep, 1+self.ep) * advantage[batch]
                weight_prob = advantage[batch] * r_t

                actor_loss = -torch.min(weight_prob, prob_clip).mean()
                actor_loss -= entropy * self.c1

                self.actor.optim.zero_grad()
                actor_loss.mean().backward()
                self.actor.optim.step()


                # critic loss
                value = self.critic(state)
                V_t1 = returns[batch] + value
                critic_loss = (V_t1 - value).pow(2)
                critic_loss = critic_loss.mean()
                self.critic.optim.zero_grad()
                critic_loss.backward()
                self.critic.optim.step()

        self.memory.clear_memory()


        


