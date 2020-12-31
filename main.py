import numpy as np
from ppo import Agent
import gym
import os
import matplotlib.pyplot as plt

def main():

    #init the env
    ENVS = ('Pendulum-v0', 'MountainCarContinuous-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2',
        'BipedalWalkerHardcore-v3')

    ENV = 'LunarLander-v2'

    # os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
    # model_dir = os.path.join(os.getcwd(), 'models')
    # os.makedirs(os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV), exist_ok=True)
    # save_dir = os.path.join(model_dir, str(datetime.date.today()) + '-' + ENV)

    env = gym.make(ENV)
    iter_per_episode = 200
    n_state = env.observation_space.shape
    n_action = env.action_space.n

    env.seed(1234)
    np.random.seed(1234)

    num_episodes = 1001

    batch_size = 12
    #Pendulum
    layer_1_nodes, layer_2_nodes = 256, 200

    GAMMA = 0.99

    ppo = Agent(n_state, n_action,layer_1_nodes=layer_1_nodes,layer_2_nodes=layer_2_nodes, batch_size=batch_size)
    n_steps = 0
    M = 10
    score = []
    for i in range(num_episodes):
        s = env.reset()
        r = 0
        done = False

        while not done:       
            # env.render()     
            prob, action, value = ppo.take_action(s)
            s_1, reward, done, _ = env.step(action)
            n_steps += 1
            r += reward
            ppo.store_memory(s, action, prob, value, reward, done)
            if n_steps % M == 0:
                ppo.train()
            s = s_1
        score.append(r)
        print(f'episode: {i}, reward: {r}, steps: {n_steps}')

    plt.plot(score)
    plt.show()

if __name__ == "__main__":
    main()