import imp
from sre_parse import State
import numpy as np
from ppo import Agent
import gym
import os
import matplotlib.pyplot as plt
import torch

from torch.utils.tensorboard import SummaryWriter

def main():

    writer = SummaryWriter()
    #init the env
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
        'BipedalWalkerHardcore-v3', 'CarRacing-v1')


    os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
    model_dir = os.path.join(os.getcwd(), 'models/')
    # save_dir = os.path.join(model_dir)
    save_dir = model_dir
    env = gym.make(ENVS[-1])
    n_state = env.observation_space.shape
    n_action = env.action_space.shape[0]

    env.seed(1234)
    np.random.seed(1234)

    max_steps = 200000
    global_steps = 0
    batch_size = 64
    #Pendulum
    layer_1_nodes, layer_2_nodes = 256, 200

    GAMMA = 0.99

    ppo = Agent(n_state, n_action,epoch=5,layer_1_nodes=layer_1_nodes,layer_2_nodes=layer_2_nodes,
                batch_size=batch_size, save_dir=save_dir, contineous=True, writer=writer)
    n_steps = 0
    M = 1000
    score = []

    #Initialize obs normalization
    s = env.reset()

    for i in range(max_steps):
        s = env.reset()
        r = 0
        done = False
        s = ppo.preprocess_image(s)

        while not done:
            # env.render()
            prob, action = ppo.take_action(s)
            s_1, reward, done, _ = env.step(action)
            s_1 = ppo.preprocess_image(s_1)
            n_steps += 1
            global_steps += 1
            r += reward
            ppo.store_memory(s, s_1, action, prob, reward, done)
            if n_steps % M == 0:
                ppo.train(global_steps)
                n_steps = 0
            s = s_1
        writer.add_scalar('run/reward_per_episode', r, global_steps)
        print(f'reward: {r}, steps: {global_steps}')
        ppo.save_models()


if __name__ == "__main__":
    main()