import numpy as np

def evaluate_policy(mdp, theta, gamma, values=None):
    """
    evaluate a policy using the bellman expectation equation
    :param mdp:
    :param theta: small value > 0
    :param gamma: discount rate
    :param values: values to update
    :return: new expected state values
    """
    values = values if values is not None else mdp.get_initial_values()
    new_values = mdp.get_initial_values()
    delta = np.inf

    while theta < delta:
        delta = 0

        for i in range(mdp.size):
            for j in range(mdp.size):
                state = (i,j)
                if mdp.is_terminal(state) or not mdp.is_valid_pos(state): continue
                old_value = values[state]
                new_value = 0

                for action in mdp.policy.get_actions(state):
                    new_state = tuple(mdp.get_new_state(state=state, action=action))
                    new_value += 1/mdp.policy.get_amount_of_actions(state) * (mdp.reward(new_state) + gamma * values[new_state])

                new_values[state] = new_value
                delta = max(abs(old_value - values[state]), delta)

        values = new_values
    return values

def improve_policy(mdp, values, gamma):
    """
    improve a policy
    :param mdp:
    :param values: values of states given by the policy evaluation
    :param gamma: discount rate
    :return: if the policy is stable
    """
    policy_stable = True

    for i in range(mdp.size):
        for j in range(mdp.size):
            state = (i,j)
            old_actions = mdp.policy.get_actions(state)
            if mdp.is_terminal(state) or not mdp.is_valid_pos(state):
                continue
            new_actions = new_policy_actions(mdp, state, values, gamma)
            mdp.policy.set_actions(state, new_actions)
            if not np.array_equal(old_actions, new_actions):
                policy_stable = False
    return policy_stable

def policy_iteration(mdp, vals=None, theta=0.1, gamma=0.8):
    vals = evaluate_policy(mdp, theta, gamma, vals)
    stable = improve_policy(mdp, vals, gamma)
    return mdp.policy, vals, stable

def value_iteration(mdp, theta, gamma):
    """
    run the value iteration algorithm on mdp
    :param mdp:
    :param theta: small value >0
    :param gamma: discount rate
    :return:
    """
    values = mdp.get_initial_values()
    delta = np.inf

    while theta < delta:
        delta = 0

        for i in range(mdp.size):
            for j in range(mdp.size):
                state = (i,j)
                if mdp.is_terminal(state) or not mdp.is_valid_pos(state):
                    continue
                old_value = values[state]
                new_value = -np.inf
                for action in mdp.policy.get_actions(state):
                    new_state = tuple(mdp.get_new_state(state=state, action=action))
                    new_value = max(new_value, (reward := mdp.reward(new_state)) + gamma * (val := values[new_state]))
                values[state] = new_value
                delta = max(abs(old_value - values[state]), delta)

    for i in range(mdp.size):
        for j in range(mdp.size):
            state = (i,j)
            if mdp.is_terminal(state) or not mdp.is_valid_pos(state):
                continue
            new_actions = new_policy_actions(mdp, state, values, gamma)
            mdp.policy.set_actions(state, new_actions)
    return values, mdp.policy

def new_policy_actions(mdp, state, values, gamma):
    best = {}
    for action in range(4):
        new_state = tuple(mdp.get_new_state(state=state, action=action))
        val = (mdp.reward(new_state) + gamma * values[new_state])
        if not best:
            best = {action: val}
        elif val > max(best.values()):
            best = {action: val}
        elif val == max(best.values()):
            best[action] = val
    return np.array(list(best.keys()))