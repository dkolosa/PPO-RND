from os import stat
import gym
from gym.envs.box2d.car_racing import CarRacing
import numpy as np
from agent import Agent
from utils import plot_learning_curve

import cProfile, pstats

if __name__ == '__main__':
    
    # env = gym.make('CartPole-v0')
    env = gym.make('CarRacing-v1')
    N = 10
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    # agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                #   alpha=alpha, n_epochs=n_epochs,
                #   input_dims=env.observation_space.shape)
    agent = Agent(n_actions=3, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    prof = cProfile.Profile()
    prof.enable()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action, prob, val = agent.choose_action(observation)
            action = action.numpy()[0]
            val = val.numpy()[0]
            prob = prob.numpy()[0]
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, observation_, action,
                                   prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        prof.disable()
        stats = pstats.Stats(prof)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats(30)
        exit()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
