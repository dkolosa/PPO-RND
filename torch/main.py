import imp
from sre_parse import State
import numpy as np
from ppo import Agent
import gym
import os
import matplotlib.pyplot as plt
import torch

def main():

    #init the env
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
        'BipedalWalkerHardcore-v3', 'CarRacing-v1')

    # ENV = 'ALE/BankHeist-v5'

    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
    model_dir = os.path.join(os.getcwd(), 'models')
    # save_dir = os.path.join(model_dir)
    save_dir = model_dir
    env = gym.make(ENVS[-1])
    iter_per_episode = 200
    n_state = env.observation_space.shape
    n_action = env.action_space.shape[0]


    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001

    batch_size = 64
    #Pendulum
    layer_1_nodes, layer_2_nodes = 256, 200

    GAMMA = 0.99

    ppo = Agent(n_state, n_action,epoch=5,layer_1_nodes=layer_1_nodes,layer_2_nodes=layer_2_nodes,
                batch_size=batch_size, save_dir=save_dir)
    n_steps = 0
    M = 5
    score = []

    #Initialize obs normalization
    s = env.reset()

    for i in range(num_episodes):
        s = env.reset()
        r = 0
        done = False
        s = ppo.preprocess_image(s)
        for _ in range(500):
            action = env.action_space.sample()
            s_1_norm, _, _, _ = env.step(action)
            s_1_norm = ppo.preprocess_image(s_1_norm)
            ppo.obs_rms.update(s_1_norm)

        while not done:
        # for _ in range(500):
            env.render()
            prob, action = ppo.take_action(s)
            s_1, reward, done, _ = env.step(action)
            s_1 = ppo.preprocess_image(s_1)
            n_steps += 1
            r += reward
            r_i = ppo.intrinsic_reward(s_1)
            ppo.store_memory(s, s_1, action, prob, reward, r_i, done)
            if n_steps % M == 0:
                ppo.train()
            s = s_1
        score.append(r)
        print(f'episode: {i}, reward: {r}, steps: {n_steps}')
        ppo.save_models()
    plt.plot(score)
    plt.show()

if __name__ == "__main__":
    main()