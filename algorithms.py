import numpy as np

import utils as utl
from environment import RaceTrack
from utils import Action, ActionValueDict, DistributionPolicy, State, StateAction, make_eps_greedy_policy

def epsilon_greedy_policy(state_action_values: ActionValueDict, epsilon: float) -> dict:
    """
    Generates an epsilon-greedy policy based on Q-values.

    :param state_action_values: The current state-action value function (Q-values)
    :param epsilon: The probability of taking a random action (exploration)
    :return: A dictionary where each state maps to a probability distribution over actions
    """
    policy = {}
    for state, action in state_action_values.items():
        
        action = next(iter(action))  # Get the single action
        policy[state] = {action: 1.0}
        
    return policy



def fv_mc_estimation(states: list[State], actions: list[Action], rewards: list[float], discount: float) -> ActionValueDict:
    visited_sa_returns = {}
    first_visit_indices = {}

    for step, (state, action) in enumerate(zip(states, actions)):
        state = tuple(state) if isinstance(state, list) else state
        state_action = (*state, action)

        # Check if this state-action pair has been encountered before
        if state_action not in first_visit_indices:
            # Record the first visit index
            first_visit_indices[state_action] = step

            # Calculate the return (G_t) starting from time step step
            discounted_return = 0
            discount_factor = 0

            for reward in rewards[step:]:
                discounted_return += (discount ** discount_factor) * reward
                discount_factor += 1

            # Store the return for the first occurrence of this state-action pair
            visited_sa_returns[state_action] = discounted_return

    return visited_sa_returns


def fv_mc_control(env: RaceTrack, epsilon: float, num_episodes: int, discount: float) -> tuple[ActionValueDict, list[float]]:
        # Runs Monte-Carlo control, using first-visit Monte-Carlo for policy evaluation and regular policy improvement

        # :param RaceTrack env: environment on which to train the agent
        # :param float epsilon: epsilon value to use for the epsilon-greedy policy
        # :param int num_episodes: number of iterations of policy evaluation + policy improvement
        # :param float discount: discount factor

        # :return visited_states_returns (dict[tuple,float]): dictionary where the keys are the unique state-action combinations visited during the episode
        # and the value of each key is the estimated discounted return of the first visitation of that key (state-action pair)
        # :return all_returns (list[float]): list of all the cumulative rewards the agent earned in each episode
    # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    returns_sum = {}
    returns_count = {}
    all_returns = []

    for episode_idx in range(num_episodes):
        # Generate an episode using the current epsilon-greedy policy
        policy = make_eps_greedy_policy(state_action_values, epsilon)
        states, actions, rewards = utl.generate_episode(policy, env)

        cumulative_return = sum(rewards)
        all_returns.append(cumulative_return)

        # Initialize return and track first-visit state-action pairs
        G = 0
        visited_state_action_pairs = set()

        # Iterate through states, actions, and rewards in reverse order
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            # Update return
            G = discount * G + reward

            # Check if state-action pair is first-visited
            if (state, action) not in visited_state_action_pairs:
                visited_state_action_pairs.add((state, action))

                # Initialize returns sum and count if necessary
                if (state, action) not in returns_sum:
                    returns_sum[(state, action)] = 0
                    returns_count[(state, action)] = 0

                # Update returns sum and count
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1

                # Update state-action value using mean return
                state_action_values[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]

    return state_action_values, all_returns


def is_mc_estimate_with_ratios(
    states: list[State],
    actions: list[Action],
    rewards: list[float],
    target_policy: DistributionPolicy,
    behaviour_policy: DistributionPolicy,
    discount: float
) -> dict[tuple[State, Action], list[tuple[float, float]]]:
    state_action_returns_and_ratios = {}

    cumulative_returns = []
    G = 0
    # Calculate cumulative returns
    for t in reversed(range(len(rewards))):
        G = rewards[t] + discount * G
        cumulative_returns.append(G)
    cumulative_returns.reverse()

    # Calculate importance sampling ratios and estimates
    importance_sampling_ratios = []
    for t in range(len(states)):
        state = tuple(states[t]) if isinstance(states[t], list) else states[t]  # Ensure state is a tuple
        action = actions[t]

        # Fetch probabilities from both policies
        target_prob = target_policy(state)[action]
        behavior_prob = behaviour_policy(state)[action]

        # Calculate the importance sampling ratio
        ratio = target_prob / behavior_prob if behavior_prob > 0 else 0
        importance_sampling_ratios.append(ratio)

    # Build the dictionary of state-action pairs and their returns and ratios
    for t in range(len(states)):
        state_action = (tuple(states[t]) if isinstance(states[t], list) else states[t], actions[t])
        
        if state_action not in state_action_returns_and_ratios:
            state_action_returns_and_ratios[state_action] = []

        # Append the cumulative return and importance sampling ratio as a tuple
        state_action_returns_and_ratios[state_action].append(
            (cumulative_returns[t], importance_sampling_ratios[t])
        )

    return state_action_returns_and_ratios




def ev_mc_off_policy_control(env: RaceTrack, behaviour_policy: DistributionPolicy, epsilon: float, num_episodes: int, discount: float):
     # Initialize memory of estimated state-action returns
    state_action_values = utl.init_q_and_v()[1]
    all_state_action_values = {}
    all_returns = []

    target_policy = utl.make_eps_greedy_policy(state_action_values, epsilon)

    for episode_idx in range(num_episodes):
        # Generate an episode using the behaviour policy
        states, actions, rewards = utl.generate_episode(utl.convert_to_sampling_policy(behaviour_policy), env)

        cumulative_return = sum(rewards)
        all_returns.append(cumulative_return)

        # Initialize importance sampling weight
        weight = 1.0

        # Iterate through states, actions, and rewards in reverse order
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            # Calculate importance sampling ratio
            target_prob = target_policy(state)
            behaviour_prob = behaviour_policy(state)[action]
            weight *= target_prob / behaviour_prob

            sa = (state, action)
            if sa not in all_state_action_values:
                all_state_action_values[sa] = []
            all_state_action_values[sa].append(weight * reward)

            state_action_values[sa] = np.mean(all_state_action_values[sa])

    return state_action_values, all_returns
