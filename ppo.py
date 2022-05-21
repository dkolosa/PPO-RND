"""This implementation is inspired from jcwleo:
https://github.com/jcwleo/random-network-distillation-pytorch
"""

import numpy as np
import torch
from model import Actor, Critic, ActorCNN, CriticCNN, RND
from torch.utils.data import BatchSampler, RandomSampler
import torch.nn.functional as F
from torch.optim import Adam


class Memory():
    def __init__(self,batch_size):
        # init state, action, reward, state_, done
        self.state = []
        self.state_1 = []
        self.prob = []
        self.action = []
        self.reward = []
        self.r_i = []
        self.done = []
        self.batch_size = batch_size

    def get_memory(self):

        return np.array(self.state),\
            np.array(self.state_1), \
            np.array(self.action),\
            np.array(self.prob),\
            np.array(self.reward),\
            np.array(self.r_i), \
            np.array(self.done)
        
    def get_batches(self):
        n_states = len(self.state)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        return [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batches)]
        
    def store_memory(self, state, s_1, action, prob, reward, r_i, done):
        self.state.append(state)
        self.state_1.append(s_1)
        self.action.append(action)
        self.reward.append(reward)
        self.r_i.append(r_i)
        self.prob.append(prob)
        self.done.append(done)

    def clear_memory(self):
        self.state.clear()
        self.state_1.clear()
        self.action.clear()
        self.reward.clear()
        self.r_i.clear()
        self.prob.clear()
        self.done.clear()
        

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count



class Agent():
    def __init__(self, num_state, num_action, epoch=5, ep=0.2, beta=3, c1=0.09, 
                layer_1_nodes=512, layer_2_nodes=256, batch_size=64,save_dir='models',
                contineous=True):
        
        self.ep = ep
        self.beta = beta
        self.c1 = c1
        self.gamma_e = .99
        self.gamma_i = .999
        self.g_lambda = 0.95

        self.r_e_coef = 2
        self.r_i_coef = 1

        self.batch_size = batch_size

        self.contineous = contineous


        self.epoch = epoch

        self.actor = ActorCNN(num_action, layer_1_nodes, layer_2_nodes, checkpt='ppo', contineous=contineous)
        self.critic = CriticCNN(layer_1_nodes, layer_2_nodes)
        
        self.rnd = RND()

        self.optimizer = Adam(list(self.actor.parameters())+list(self.critic.critic_ext.parameters()) +\
                              list(self.critic.critic_int.parameters()) +\
                              list(self.rnd.predictor.parameters()),
                              lr=1e-4)

        self.optimizer.param_groups

        self.obs_rms = RunningMeanStd()
        self.rwd_rms = RunningMeanStd(shape=(1,3,86,86))

        self.memory = Memory(batch_size)

        self.save_dir = save_dir


    

    def take_action(self,state):
        with torch.no_grad():

            state = torch.tensor([state], dtype=torch.float, device=self.actor.device)
            # torch.permute(state, (2,0,1))
            prob_dist = self.actor(state)
            action = prob_dist.sample()
            action = torch.clamp(action, -1, 1)
            # action = tytorch.clamp(action, -1, 1)
            prob = torch.squeeze(prob_dist.log_prob(action)).cpu().detach().numpy()
            action = torch.squeeze(action).cpu().detach().numpy()


        return prob, action

    def preprocess_image(self,image):
        # pytorch image: C x H x W
        image = image[0:86, 0:86, 0:3]
        image_swp = np.swapaxes(image, -1, 0)
        image_swp = np.swapaxes(image_swp,-1, -2)
        return image_swp/255.0

    def store_memory(self, state, s_1, action, prob, reward, r_i, done):
        self.memory.store_memory(state, s_1, action, prob, reward, r_i, done)

    def calculate_adv_ret(self, state, state_1, reward, done, ext=True):
        with torch.no_grad():
            if ext:
                value, _ = self.critic(state)
                value_1, _ = self.critic(state_1)
                gamma = self.gamma_e
            else:
                _, value = self.critic(state)
                _, value_1 = self.critic(state_1)
                gamma = self.gamma_i
            
            
            delta = reward + gamma * value_1 - value
            delta = delta.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(delta[::-1], done[::-1]):
                advantage = dlt + gamma * self.g_lambda * adv[-1] * \
                            (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = torch.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            returns = adv + value
            adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def intrinsic_reward(self, state): 
        # Normalize the next_state obs
        stae = np.clip((state - self.obs_rms.mean)/ self.obs_rms.var, -5, 5)
        state = torch.tensor([state], dtype=torch.float, device=self.rnd.device)
        predict, target = self.rnd(state)
        int_reward = (predict - target).pow(2).mean(1).cpu().detach().numpy()
        return int_reward

    def train(self):
        epochs = 5
            
        state_mem, state_1_mem, action_mem, prob_mem, reward_mem, r_i_mem, done_mem = self.memory.get_memory()
        states = torch.tensor(state_mem, dtype=torch.float, device=self.actor.device)
        states_1 = torch.tensor(state_1_mem, dtype=torch.float, device=self.actor.device)
        actions = torch.tensor(action_mem, dtype=torch.float, device=self.actor.device)
        old_probs = torch.tensor(prob_mem, dtype=torch.float, device=self.actor.device)
        rewards = torch.tensor(reward_mem, dtype=torch.float, device=self.actor.device)
        r_i = torch.tensor(r_i_mem, dtype=torch.float, device=self.actor.device)


        advantage, returns = self.calculate_adv_ret(states, states_1, rewards, done_mem, ext=True)
        advantage_int, returns_int = self.calculate_adv_ret(states, states_1, r_i, done_mem, ext=False)

        advantage = advantage + advantage_int

        forward_mse = torch.nn.MSELoss(reduction='none')

        for epoch in range(epochs):
            # batches = self.memory.get_batches()
            for batch in BatchSampler(RandomSampler(range(len(self.memory.state))), self.batch_size, False):
            # for batch in batches:

                # calculate r_t(theta)
                state = states[batch]
                old_prob = old_probs[batch]
                action = actions[batch]

                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(states_1[batch])
                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.rnd.device)
                mask = (mask < 0.25).type(torch.FloatTensor).to(self.rnd.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.rnd.device))

                dist_new = self.actor(states)
                entropy = dist_new.entropy().sum(1, keepdims=True)
                prob_new = dist_new.log_prob(action)

                r_t = (prob_new.sum(1, keepdims=True) - old_prob.sum(1, keepdims=True)).exp()        

                # L_clip
                prob_clip = torch.clamp(r_t, 1-self.ep, 1+self.ep) * advantage[batch]
                weight_prob = advantage[batch] * r_t

                actor_loss = -torch.min(weight_prob, prob_clip)
                actor_loss = (actor_loss - entropy * self.c1).mean()

                # self.actor.optim.zero_grad()
                # actor_loss.mean().backward()
                # self.actor.optim.step()


                # critic loss
                value_ext, value_int = self.critic(state)
                
                critic_ext_loss = F.mse_loss(returns[batch], value_ext)
                critic_int_loss = F.mse_loss(returns_int[batch], value_int)

                critic_loss = critic_ext_loss + critic_int_loss

                # self.critic.optim.zero_grad()
                # critic_loss.backward()
                # self.critic.optim.step()

                total_loss = 0.5 * critic_loss + actor_loss + forward_loss
                total_loss.backward()
                # self.global_grad_norm_(list(self.actor.parameters())+list(self.critic.critic_ext.parameters()) +\
                #                   list(self.critic.critic_int.parameters()) +\
                #                   list(self.rnd.predictor.parameters()))
                self.optimizer.step()

        self.memory.clear_memory()

    def save_models(self):
        print(f'Saving Models')
        torch.save(self.actor.state_dict(), self.save_dir+'_actor.ckpt')
        torch.save(self.critic.state_dict(), self.save_dir+'_critic.ckpt')
        torch.save(self.rnd.state_dict(), self.save_dir+'_rnd.ckpt')


    def global_grad_norm_(self, parameters, norm_type=2):
        r"""Clips gradient norm of an iterable of parameters.
        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.
        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        norm_type = float(norm_type)
        if norm_type == inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)

        return total_norm

        


