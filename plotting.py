import matplotlib.pyplot as plt
from seaborn import heatmap

def plot(values, policy, title=""):
    plt.figure(figsize=(6,6))
    heatmap(values.T, annot=True, cmap="Spectral", linewidths=0.5, square=True, cbar=True,
            annot_kws={'color':'w', 'alpha':1})

    grid_size = values.shape[0]
    action_arrows = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }
    for i in range(grid_size):
        for j in range(grid_size):
            for action, (dx, dy) in action_arrows.items():
                if action in policy.get_actions((j,i)):
                    plt.quiver(j+0.5, i+0.5, dx, dy, angles="xy", scale_units="xy", width=0.01)

    plt.title(title)
    plt.show()

def plot_steps(x, p_steps, v_steps):
    plt.figure(figsize=(6, 6))
    plt.plot(x, p_steps, label='policy iteration steps')
    plt.plot(x, v_steps, label='value iteration steps')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('discount factor')
    plt.ylabel('amount of steps')
    plt.title("amount of steps in episode, following policy")
    plt.xticks([x / 10 for x in range(0, 11)])
    plt.show()