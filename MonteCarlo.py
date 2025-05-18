# %%
import numpy as np
import random
from collections import defaultdict
from Policy import Policy
import matplotlib.pyplot as plt


# %%
def generate_episode(env, policy, max_steps=100):
    """
    Generate an episode using a policy acting in a simulation.

    Args:
        env: The environment to interact with.
        policy: The policy used to select actions.
        max_steps: Maximum number of steps in the episode.

    Returns:
        states: List of visited states (excluding the initial one).
        actions: List of actions taken.
        rewards: List of rewards received.
    """
    states, actions, rewards = [], [], []

    # Exploring starts: begin from a random state and take a random action
    obs, _ = env.reset_random()
    first_action = random.choice([0, 1, 2, 3])
    new_obs, reward, done, _, _ = env.step(first_action)

    states.append(obs)
    actions.append(first_action)
    rewards.append(reward)
    obs = new_obs

    # Continue episode until max_steps or terminal state
    for _ in range(max_steps - 1):
        probabilities = policy.get_action_probabilities(obs)
        action = np.random.choice(len(probabilities), p=probabilities)
        obs, reward, done, _, _ = env.step(action)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        # if the terminal state is arrived stop the espisode
        if done:
            break

    # Remove the initial state (used only for exploring start)
    states.pop(0)
    return states, actions, rewards


# %%
def monte_carlo_es(env, gamma=0.7, epsilon=0.1, num_episodes=4000):
    """
    Monte Carlo Exploring Starts algorithm with greedy policy updates.

    Args:
        env: The environment.
        gamma: Discount factor.
        epsilon: Starting epsilon (decayed but not used in policy here).
        num_episodes: Number of episodes to run.

    Returns:
        Q: Estimated action-value function.
        policy: Improved policy.
    """
    # All valid grid positions (states)
    states = [(i, j) for i in range(env.size) for j in range(env.size) if env.is_valid_pos((i, j))]

    # Initialize random policy and action-value function
    policy = Policy.get_random_policy_uniform(env.size)
    returns = {(tuple(state), action): [] for state in states for action in range(env.action_space.n)}
    Q = {(tuple(state), a): np.random.uniform(-1, 1) for state in states for a in range(env.action_space.n)}
    cumulative_rewards = []

    for _ in range(num_episodes):
        epsilon = max(0.01, epsilon * 0.99)  # Decaying epsilon (not used here though)
        states, actions, rewards = generate_episode(env, policy)
        g = 0
        episode_reward = sum(rewards)
        cumulative_rewards.append(episode_reward)
        visited = set()

        # Loop through episode in reverse to calculate returns
        for t in reversed(range(len(states))):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            g = gamma * g + reward
            state_action_pair = (tuple(state), action)

            if state_action_pair not in visited:
                visited.add(state_action_pair)
                returns[state_action_pair].append(g)
                Q[state_action_pair] = np.mean(returns[state_action_pair])

                # Policy improvement (greedy)
                q_values = [Q[(tuple(state), a)] for a in range(env.action_space.n)]
                best_action = np.argmax(q_values)
                policy.set_actions(state, [best_action])

    return Q, policy, cumulative_rewards


# %%
def monte_carlo_epsilon_soft(env, gamma=0.7, epsilon=0.8, num_episodes=4000):
    """
    Monte Carlo Exploring Starts algorithm with epsilon-soft policy updates.

    Args:
        env: The environment.
        gamma: Discount factor.
        epsilon: Starting epsilon (decayed and used in policy).
        num_episodes: Number of episodes to run.

    Returns:
        Q: Estimated action-value function.
        policy: Improved epsilon-soft policy.
    """
    # All valid grid positions (states)
    states = [(i, j) for i in range(env.size) for j in range(env.size) if env.is_valid_pos((i, j))]

    # Initialize random policy and action-value function
    policy = Policy.get_random_policy_uniform(env.size)
    returns = {(tuple(state), action): [] for state in states for action in range(env.action_space.n)}
    Q = {(tuple(state), a): np.random.uniform(-1, 1) for state in states for a in range(env.action_space.n)}
    cumulative_rewards = []

    for _ in range(num_episodes):
        epsilon = max(0.01, epsilon * 0.99)
        states, actions, rewards = generate_episode(env, policy)
        g = 0
        episode_reward = sum(rewards)
        cumulative_rewards.append(episode_reward)
        visited = set()

        # Loop through episode in reverse to calculate returns
        for t in reversed(range(len(states))):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            g = gamma * g + reward
            state_action_pair = (tuple(state), action)

            if state_action_pair not in visited:
                visited.add(state_action_pair)
                returns[state_action_pair].append(g)
                Q[state_action_pair] = np.mean(returns[state_action_pair])

                # Policy improvement (epsilon-soft)
                q_values = [Q[(tuple(state), a)] for a in range(env.action_space.n)]
                best_action = np.argmax(q_values)
                probs = np.ones(env.action_space.n) * epsilon / env.action_space.n
                probs[best_action] += 1 - epsilon
                policy._policy[state[0], state[1], :] = probs

    return Q, policy, cumulative_rewards
