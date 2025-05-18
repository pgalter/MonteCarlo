import numpy as np
from env import CustomEnv
import random as rnd
from env import CustomEnv, Actions


def TD_0(policy, gamma=0.9):
    alpha = 0.1
    env = CustomEnv()
    V = env.get_initial_values()

    episode_rewards = []
    episodes = 100

    for i in range(episodes):
        S, _ = env.reset()
        episode_reward = 0
        v=0
        for t in range(1000):
            A = policy.sample(tuple(S))
            obs, reward, done, _, info = env.step(A)
            V[tuple(S)] += alpha * (reward + 0.9 * V[tuple(obs)] - V[tuple(S)])
            episode_reward += reward
            if done:
                break
            S = obs

        episode_rewards.append(episode_reward)

    return episode_rewards






def epsilon_greedy_policy_sarsa(Q, state, eps):
    if np.random.random() < eps:
        return np.random.choice(list(Actions.items()))
    else:
        return np.argmax(Q[tuple(state)])


def Sarsa(env, episodes, alpha, gamma, eps):

    Q = np.zeros((env.size, env.size, len(Actions.items())))
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy_policy_sarsa(Q, state, eps)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy_policy_sarsa(Q, next_state, eps)
            Q[tuple(state)][action] += alpha * (reward + gamma * Q[tuple(next_state)][next_action] - Q[tuple(state)][action])
            state = next_state
            action = next_action

            episode_reward += reward

        episode_rewards.append(episode_reward)



    return episode_rewards


def epsilon_greedy_policy_Q(Q, state, eps):
    if np.random.random() < eps:
        return np.random.choice(Actions.items())
    else:
        return np.argmax(Q[state[0], state[1], :])

def Q_learning(env, episodes, alpha, gamma, eps):

    Q = np.zeros((env.size, env.size, len(Actions.items())))
    episode_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            action = epsilon_greedy_policy_Q(Q, state, eps)
            next_state, reward, done, _, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], best_next_action] - Q[state[0], state[1], action])

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
    return episode_rewards