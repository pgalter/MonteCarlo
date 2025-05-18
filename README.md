# MonteCarlo
Policy Optimization with ε-Soft Monte Carlo in a MDP Environment 

This was a collaborative project in which I was responsible for the policy optimiziation with normal Monte Carlo and ε-Soft Monte Carlo.

This project focuses on evaluating and comparing different reinforcement learning (RL) algo-
rithms using a custom-built environment designed to simulate an underwater world. The agent,
represented by the character ”Nemo,” must navigate a grid-based ocean environment to reach
its goal while avoiding obstacles.

The environment consists of an 8x8 grid, following the Markov Decision Process (MDP)
properties. Nemo starts at a predefined position and must reach the goal state, represented
by an anemone, to terminate the episode. Along the way, various obstacles such as seaweed,
representing impassable walls, marine creatures, and pollution influence the agent’s decision-
making process by imposing negative rewards.
Rewards are designed to guide the agent towards optimal behavior, with immediate rewards
influencing short-term decisions, while cumulative rewards determine long-term strategy opti-
mization.

Additionally, the environment is a deterministic process, meaning that a given action in a
state always leads to the same next state unless blocked by an obstacle.
Nemo can take actions from a discrete action space of size 4, corresponding to movement
in four cardinal directions. Each action leads to a new state, and a reward is given. In the
outermost grid states, any action that would move the agent beyond the grid boundaries instead
results in the agent remaining in its current position.
The primary objective is to train the agent to efficiently navigate to the goal while avoiding
hazards and minimizing negative rewards. The agent should learn an optimal policy that
maximizes cumulative rewards. A small movement penalty is implemented to encourage shorter
paths towards the goal.

![image](https://github.com/user-attachments/assets/cb700309-0654-49dd-8c41-a678953b817b)



# My results: 
![image](https://github.com/user-attachments/assets/4ff354ea-7821-479b-85c2-88f093d2e564)
The cumulative reward of the pure greedy version shows a variant and somewhat upward
trend over the 4000 episodes, indicating that the algorithm gradually learns better policies.
This aligns with our expectations, seen as Monte Carlo methods require many episodes to
converge because they rely on sample returns rather than bootstrapping. The high variation
can be attributed to its lack of exploration. It is possible that the algorithm overfits too early
and sticks to incomplete experiences.

In contrast, the epsilon-soft version outperforms the pure greedy version: it increases more
exponentially and consistently. This shows that our environment first requires a lot of explo-
ration to avoid getting stuck in local optima due to the fact that rewards are scarce.
