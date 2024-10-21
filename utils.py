from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from environment import RaceTrack


State = tuple[int, int, int, int]
Action = int
StateAction = tuple[int, int, int, int, Action]
ValueDict = dict[State, float]
ActionValueDict = dict[StateAction, float]
Policy = Callable[[State], int]
DistributionPolicy = Callable[[State], list[float]]



def plot_many(experiments: list[list[float]], label: Optional[str] = None, color: Optional[str] = None) -> None:
    mean_exp = np.mean(experiments, axis=0)
    std_exp = np.std(experiments, axis=0)
    plt.plot(mean_exp, color=color, label=label)
    plt.fill_between(range(len(experiments[0])), mean_exp + std_exp, mean_exp - std_exp, color=color, alpha=0.1)
    plt.show()


def init_q_and_v() -> tuple[list[StateAction], ActionValueDict, list[State], ValueDict]:
    state_action_array = list(product(np.arange(12), np.arange(10),
                                      np.arange(-4, 1, 1), np.arange(-4, 5, 1), np.arange(0, 9, 1)))
    state_array = list(product(np.arange(12), np.arange(10),
                               np.arange(-4, 1, 1), np.arange(-4, 5, 1)))
    state_action_returns = {}
    state_returns = {}
    for sa in state_action_array:
        state_action_returns[sa] = 0.
    for s in state_array:
        state_returns[s] = 0.
    return state_action_array, state_action_returns, state_array, state_returns


def qs_from_q(state_action_values: ActionValueDict = None) -> dict[State, list[float]]:
    all_states = init_q_and_v()[2]
    state_values = {state: np.zeros(9) for state in all_states}
    
    for sa in state_action_values:
        if sa[:-1] in state_values:  # Check if state exists
            state_values[sa[:-1]][sa[-1]] = state_action_values[sa]
    
    return state_values


def random_argmax(value_list: list) -> np.ndarray:
    """ a random tie-breaking argmax """
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def make_eps_greedy_policy(state_action_values: ActionValueDict, epsilon: float) -> Policy:
    n_actions = 9
    state_values = qs_from_q(state_action_values)

    def policy(state: State) -> Action:
        action_values = [state_action_values.get((state, action), 0) for action in range(n_actions)]

        if np.random.random() < epsilon:
            action = np.random.choice(range(n_actions))
        else:
            max_value = max(action_values)
            best_actions = [i for i, value in enumerate(action_values) if value == max_value]
            action = np.random.choice(best_actions)
        
        return action

    return policy


def generate_episode(policy: Policy, env: RaceTrack) -> tuple[list[State], list[Action], list[float]]:
    states = []
    rewards = []
    actions = []

    state = env.reset()
    done = False  # To check if the episode is done

    while not done:
        states.append(state)

        action = policy(state)

        actions.append(action)

        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)

        state = next_state

    return states, actions, rewards


def make_eps_greedy_policy_distribution(state_action_values: ActionValueDict, epsilon: float) -> DistributionPolicy:
    n_actions = 9
    state_values = qs_from_q(state_action_values)

    def policy(state: State) -> list[float]:
        action_probabilities = np.zeros(n_actions)
        q_values = [state_action_values.get((state, a), 0) for a in range(n_actions)]
        max_q_value = max(q_values)

        for a in range(n_actions):
            if q_values[a] == max_q_value:
                action_probabilities[a] = (1 - epsilon) / len([q for q in q_values if q == max_q_value]) + epsilon / n_actions
            else:
                action_probabilities[a] = epsilon / n_actions

        return action_probabilities

    return policy


def convert_to_sampling_policy(distribution_policy: DistributionPolicy) -> Policy:
    def policy(state):
        action_probabilities = distribution_policy(state)
        return np.random.choice(len(action_probabilities), p=action_probabilities)

    return policy
