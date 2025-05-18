#%%
import gymnasium as gym
from env import CustomEnv
from plotting import plot, plot_steps
import DynamicProgramming
import MonteCarlo
import TemporalDifference
import matplotlib.pyplot as plt
from Policy import Policy
import numpy as np
import pandas as pd

def render_episode(env, render_mode):
    env.set_render_mode(render_mode)
    env.reset(pos_only=True)
    total_reward = 0
    total_steps = 0
    while True:
        env.render()
        obs, reward, done, truncated, info = env.step()
        total_reward += reward
        total_steps += 1
        if CustomEnv.was_closed():
            done = True
        if done:
            break
    env.render()
    return total_reward, total_steps
#%%
# ------- dynamic programming ------
def policy_iteration(env, theta, gamma):
    env.reset()
    values = None
    stable = False
    policy = env.policy
    while not stable:
        policy, values, stable = DynamicProgramming.policy_iteration(env, values, theta, gamma)
    reward, steps = render_episode(env, 'sim')
    return reward, steps, values, policy

def value_iteration(env, theta, gamma):
    env.reset()
    values, policy = DynamicProgramming.value_iteration(env, theta, gamma)
    reward, steps = render_episode(env, 'sim')
    return reward, steps, values, policy

def show_dp_results(env):
    p_steps = []
    v_steps = []
    values = None
    x = [x / 100 for x in range(0, 100, 5)]
    for gamma in x:
        reward, steps, values, policy = value_iteration(env, 0.1, gamma)
        v_steps.append(steps)

        reward, steps, values, policy = policy_iteration(env, 0.1, gamma)
        p_steps.append(steps)

    print("Results ready, plotting:")
    plot_steps(x, p_steps, v_steps)
    plot(values, env.policy, "policy iteration - values & policy, gamma=0.95")
    render_episode(env, 'human')
    render_episode(env, 'human')
    render_episode(env, 'human')

#%%
# ------- monte carlo ---------

def monte_carlo_es_plot(rewards):

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(rewards)), rewards, label='Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Monte Carlo (not ε-Soft) Learning Progress')
    plt.legend()
    smoothed = pd.Series(rewards).rolling(50).mean()
    plt.plot(smoothed)
    plt.show()

def monte_carlo_epsilon_soft_plot(rewards):

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(rewards)), rewards, label='Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Monte Carlo ε-Soft Learning Progress')
    plt.legend()
    smoothed = pd.Series(rewards).rolling(50).mean()
    plt.plot(smoothed)
    plt.show()

def plot_learning_curve(cumulative_rewards, xlabel="Episodes", ylabel="Cumulative Reward", e_soft=False):
    """
    Plot the learning curve using cumulative rewards.

    Args:
        cumulative_rewards: List of cumulative rewards per episode.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """
    plt.plot(cumulative_rewards)
    if e_soft:
        plt.title("Learning Curve Monte Carlo ε-Soft")
    else:
        plt.title("Learning Curve Monte Carlo (not ε-Soft)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def smooth_rewards(cumulative_rewards, window_size=100):
    """
    Apply a moving average to smooth out the reward curve.

    Args:
        cumulative_rewards: List of cumulative rewards.
        window_size: Size of the moving average window.

    Returns:
        Smoothed rewards.
    """
    return np.convolve(cumulative_rewards, np.ones(window_size) / window_size, mode='valid')

# ------- temporal difference -------
def show_TD_0(policy):
    all_rewards = {}
    gammas = [x / 100 for x in range(0, 100, 5)]

    for gamma in gammas:
        print(f"Running TD(0) with gamma={gamma}")
        rewards = TemporalDifference.TD_0(policy, gamma=gamma)
        cumulative = [sum(rewards[:i+1]) for i in range(len(rewards))]
        all_rewards[gamma] = cumulative

    # Plotting
    plt.figure(figsize=(10, 6))
    for gamma, cumulative_rewards in all_rewards.items():
        plt.plot(cumulative_rewards, label=f'γ = {gamma}')

    plt.title('Cumulative Rewards over Episodes for Different Gamma Values')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Sarsa TESTLOOP


def Sarsa_test(env):
    all_rewards = {}
    gamma_vals = [x / 100 for x in range(0, 100, 5)]
    for gamma in gamma_vals:
        print(f"Running Sarsa with γ = {gamma}")
        rewards = TemporalDifference.Sarsa(env, episodes=1000, alpha=0.1, gamma=0.9, eps=0.1)
        cumulative = [sum(rewards[:i + 1]) for i in range(len(rewards))]
        all_rewards[gamma] = cumulative

    # Plot
    plt.figure(figsize=(10, 6))
    for gamma, cumulative_rewards in all_rewards.items():
        plt.plot(cumulative_rewards,  label=f'γ = {gamma}')

    plt.title(f'Cumulative Rewards over Episodes Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Qlearning testloop


def Q_learning_test(env):
    all_rewards = {}
    gamma_vals = [x / 100 for x in range(0, 100, 5)]
    for gamma in gamma_vals:
        print(f"Running Sarsa with γ = {gamma}")
        rewards = TemporalDifference.Q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, eps=0.1)
        cumulative = [sum(rewards[:i + 1]) for i in range(len(rewards))]
        all_rewards[gamma] = cumulative

    # Plot
    plt.figure(figsize=(10, 6))
    for gamma, cumulative_rewards in all_rewards.items():
        plt.plot(cumulative_rewards,  label=f'γ = {gamma}')

    plt.title(f'Cumulative Rewards over Episodes Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def main():
    # Register the environment
    print("Creating the environment...")
    from gymnasium.envs.registration import register
    register(
        id='CustomEnv-v0',
        entry_point='__main__:CustomEnv'
    )
    env = gym.make('CustomEnv-v0', render_mode="sim", movement_penalty=0.5).env.env

    print("generating Dynamic Programming results...")
    show_dp_results(env)
    env.close()

    print("generating Monte Carlo results:")
    _, _, rewards = MonteCarlo.monte_carlo_es(env)
    smoothed_rewards = smooth_rewards(rewards, window_size=200)
    plot_learning_curve(smoothed_rewards, e_soft=False)
    monte_carlo_es_plot(rewards)

    _, _, rewards = MonteCarlo.monte_carlo_epsilon_soft(env)
    smoothed_rewards = smooth_rewards(rewards, window_size=200)
    plot_learning_curve(smoothed_rewards, e_soft=True)
    monte_carlo_epsilon_soft_plot(rewards)

    env.close()

    policy = Policy.get_random_policy(8)
    show_TD_0(policy)

    Sarsa_test(env)
    Q_learning_test(env)
    Q = MonteCarlo.monte_carlo_es(env)
    print(Q)
    
    env.close()

if __name__=="__main__":
    main()
