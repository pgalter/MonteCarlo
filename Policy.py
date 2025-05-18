#%%
import numpy as np
#%%
class Policy:
    def __init__(self, policy):
        self._policy = policy

    def sample(self, state: tuple) -> int:
        """
        sample a random action within the policy at state
        :param state: a tuple of x,y position of nemo
        :return: a random action within policy, can be 0,1,2,3
        """
        options = np.flatnonzero(self._policy[state])
        return np.random.choice(options)

    def get_actions(self, state: tuple) -> np.array:
        """
        return all the actions in a policy at state
        :param state: a tuple of x,y, position of nemo
        :return: array of actions within policy at state: for example [0,3]
        """
        return np.flatnonzero(self._policy[state])

    def set_actions(self, state: tuple, actions) -> None:
        """
        set the actions in the policy at state
        :param state: tuple of x,y, position of nemo
        :param actions: actions that should be in the policy, for example: [0,2]
        """
        a = np.zeros(4)
        for act in actions:
            a[act] = 1/len(actions)
        self._policy[state] = a

    def get_amount_of_actions(self, state: tuple) -> int:
        """
        return the number of actions in the policy at state
        :param state: tuple of x,y, position of nemo
        :return: number of actions
        """
        return np.count_nonzero(self._policy[state])

    @staticmethod
    def get_random_policy(size):
        """
        return a uniform random policy
        :param size: size of grid
        :return: Policy
        """
        return Policy(np.ones([size, size, 4])) 

    def get_random_policy_uniform(size):
        """
        return a uniform random policy
        :param size: size of grid
        :return: Policy
        """
        return Policy(np.ones([size, size, 4])/4) 

    def best_action(self, state: tuple) -> int:
        """
        Obtains the best action for a given state based on the highest probability.
        
        :param state: a tuple of x,y position of nemo
        :return: the best action (0,1,2,3) with the highest probability in policy
        """
        
        return np.argmax(self._policy[state])

    def random_sample(self, state: tuple) -> int:
        """
        Sample a random action within the policy at state.
        :param state: a tuple of x, y position of nemo
        :return: a random action within policy, can be 0, 1, 2, 3
        """
        return np.random.choice(range(4), p=self._policy[state[0], state[1], :])
    
    def greedy_action(self, state: tuple) -> int:
        """
        Select the greedy (highest-probability) action at the given state.
        :param state: (x, y) position
        :return: action with max probability
        """
        return int(np.argmax(self._policy[state[0], state[1], :]))

    def get_action_probabilities(self, state: tuple) -> np.ndarray:
        """
        Get the normalized action probabilities at a given state.

        :param state: (x, y) position
        :return: np.array of shape (4,) summing to 1
        """
        probs = self._policy[state[0], state[1], :]

        total = np.sum(probs)
        if total == 0:
            # Fall back to uniform distribution (or any default strategy)
            return np.ones(4) / 4
        
        return probs / total
# %%
