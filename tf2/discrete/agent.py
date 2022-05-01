from dis import dis
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions
from model import ActorCritic

class Memory():
    def __init__(self,batch_size):
        # init state, action, reward, state_, done
        self.state = []
        self.action = []
        self.reward = []
        self.val = []
        self.prob = []
        self.done = []
        self.batch_size = batch_size

    def get_memory(self):

        self.n_states = len(self.state)
        batch_st = np.arange(0, self.n_states, self.batch_size)
        idx = np.arange(self.n_states, dtype=np.int16)
        np.random.shuffle(idx)

        batches = [idx[i:i+self.batch_size] for i in batch_st]

        return np.array(self.state),\
            np.array(self.action),\
            np.array(self.reward),\
            np.array(self.val),\
            np.array(self.prob),\
            np.array(self.done),\
            batches
        
    def store_memory(self, state, action, reward, val, prob, done):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.val.append(val)
        self.prob.append(prob)
        self.done.append(done)

    def clear_memory(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.val.clear()
        self.prob.clear()
        self.done.clear()
        

class Agent():
    def __init__(self, num_state, num_actions, ep=0.2, beta=3, c1=0.1, layer_1_nodes=512, layer_2_nodes=256, batch_size=64,save_dir='models'):
        
        self.ep = ep
        self.beta = beta
        self.c1 = c1
        self.gamma = .99
        self.g_lambda = 0.95
        self.epochs = 100

        self.batch_size = batch_size

        self.num_actions = num_actions

        # self.actor = Actor(num_state, num_action, layer_1_nodes, layer_2_nodes)
        # self.critic = Critic(num_state, layer_1_nodes, layer_2_nodes)

        self.model = ActorCritic(num_state, num_actions)

        self.model.actor.compile(optimizer='adam')
        self.model.critic.compile(optimizer='adam')

        self.memory = Memory(batch_size)

        self.save_dir = save_dir

    @tf.function
    def take_action(self,state):
        state = tf.convert_to_tensor([state])

        prob_dist, value = self.model(state)

        dist = distributions.Categorical(prob_dist)
        action = dist.sample(self.num_actions)
        prob = dist.log_prob(action)

        return action, prob, value

    def store_memory(self, state,action, prob, val, reward, done):
        self.memory.store_memory(state, action, reward, val, prob, done)

    def save_model(self):
        self.model.actor.save(self.save_dir + '_act.ckpt')
        self.model.critic.save(self.save_dir + '_crit.ckpt')

    def train(self):

        for _ in range(self.epochs):
            state_mem, action_mem, reward_mem, val_mem, prob_mem, done_mem, batches = self.memory.get_memory()

            # Calcualte the advantage
            advan = np.zeros(len(reward_mem), dtype=np.float32)
            values = val_mem

            for T in range(len(reward_mem)-1):
                a_t = 0
                discount = 1
                for k in range(T, len(reward_mem)-1):
                    a_t += discount * (reward_mem[k] + self.gamma * val_mem[k+1]*(1-done_mem[k]) \
                        - val_mem[k])
                    discount *= self.gamma * self.g_lambda
                advan[T] = a_t 
            
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:

                    states = tf.convert_to_tensor(state_mem[batch], dtype=tf.float32)
                    old_prob = tf.convert_to_tensor(prob_mem[batch], dtype=tf.float32)
                    actions = tf.convert_to_tensor(action_mem[batch], dtype=tf.float32)
                    
                    # calculate r_t(theta)
                    policy, value = self.model(states)

                    dist_new = distributions.Categorical(policy)
                    prob_new = dist_new.log_prob(actions)

                    value = tf.squeeze(value, 1)

                    ratio = tf.math.exp(prob_new - old_prob)
                    weight_prob = advan[batch] * ratio

                    # L_clip
                    prob_clip = tf.clip_by_value(ratio, 1-self.ep, 1+self.ep) * advan[batch]
                    actor_loss = tf.reduce_mean(-tf.math.minimum(weight_prob, prob_clip))

                    # critic loss
                    V_t1 = advan[batch] + val_mem[batch]
                    critic_loss = tf.keras.losses.MSE(value, V_t1)

                actor_params = self.model.actor.trainable_variables
                critic_params = self.model.critic.trainable_variables
                
                grad_act = tape.gradient(actor_loss, actor_params)
                grad_crit = tape.gradient(critic_loss, critic_params)

                self.model.actor.optimizer.apply_gradients(zip(grad_act, actor_params))
                self.model.critic.optimizer.apply_gradients(zip(grad_crit, critic_params))


        self.memory.clear_memory()
        self.save_model()
        self.save_model()