from sre_parse import State
from turtle import done
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from memory import PPOMemory
from networks import ActorCriticCNN, ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, n_actions, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        # self.actor = ActorNetwork(n_actions)
        # self.actor.compile(optimizer=Adam(learning_rate=alpha))
        # self.critic = CriticNetwork()
        # self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)

        self.model = ActorCriticCNN(n_actions)
        self.model.actor.compile(optimizer='adam')
        self.model.critic.compile(optimizer='adam')

    def store_transition(self, state, state_, action, probs, vals, reward, done):
        self.memory.store_memory(state, state_, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')

    @tf.function
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        # probs = self.actor(state)
        probs, value = self.model(state)
        # dist = tfp.distributions.Categorical(probs)
        dist = tfp.distributions.Normal(probs, 1)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # value = self.critic(state)

        return action, log_prob, value

    def calculate_advantage_returns(self, values, r, dones):

        # _ values = self.model(states)
        _, values_ = self.model(new_states)
        deltas = r + self.gamma * values_ - values
        deltas = deltas.numpy()[0]
        adv = [0]
        for dlt, mask in zip(deltas[::-1], dones[::-1]):
            advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * \
                        (1 - mask)
            adv.append(advantage)
        adv.reverse()
        adv = adv[:-1]
        adv = np.array(adv)
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std()+1e-4)
        return adv, returns

    def calc_advantage(self, r, values, dones):
        advantage = np.zeros(len(r), dtype=np.float32)
        for t in range(len(r)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(r)-1):
                a_t += discount*(r[k] + self.gamma*values[k+1] * (
                    1-int(dones[k])) - values[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        returns = advantage + values
        return advantage


    def learn(self):


        state_arr, state_arr_, action_arr, old_prob_arr, vals_arr,\
        reward_arr, dones_arr, = self.memory.get_mems()

        advantage = self.calc_advantage(reward_arr, vals_arr, dones_arr)  

        for _ in range(self.n_epochs):

            batches = self.memory.generate_batches()
            values = vals_arr

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch], dtype=tf.float32)
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch],  dtype=tf.float32)
                    actions = tf.convert_to_tensor(action_arr[batch],  dtype=tf.float32)

                    probs, critic_value = self.model(states)
                    # dist = tfp.distributions.Categorical(probs)
                    dist = tfp.distributions.Normal(probs, 1)
                    new_probs = dist.log_prob(actions)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)

                    weighted_probs = np.reshape(advantage[batch],(prob_ratio.shape[0],1)) * prob_ratio
                    # weighted_probs = advantage[batch] * prob_ratio

                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * tf.reshape(advantage[batch], shape=(clipped_probs.shape[0],1))
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)

                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = tf.math.reduce_mean(tf.math.pow(
                                                     returns-critic_value, 2))
                    # critic_loss = keras.losses.MSE(critic_value, returns)

                actor_params = self.model.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.model.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.model.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.model.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))
        self.memory.clear_memory()
