from typing import Any
from environment import RaceTrack
import utils as utl
from utils import Action, Policy, State
import numpy as np
import random


class Agent():
    def __init__(self):
        pass

    def agent_init(self, agent_init_info: dict[str, Any]):
        """
            Setup for the agent called when the experiment first starts.

            :param dict[str,Any] agent_init_info: dictionary of parameters used to initialize the agent. The dictionary must contain the following fields:

            >>> {
                seed (int): The seed to use to initialize randomness,
                env (RaceTrack): The environment to train on,
                epsilon (float): The epsilon parameter for exploration,
                step_size (float): The learning rate alpha for the TD updates,
                discount (float): The discount factor,
            }
        """
        np.random.seed(agent_init_info['seed'])
        random.seed(agent_init_info['seed'])
        # Store the parameters provided in agent_init_info.
        self.env = agent_init_info["env"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]

        # Create a dictionary for action-value estimates and initialize it to zero.
        self.sa_array, self.q, _1, _2 = utl.init_q_and_v()

    def get_current_policy(self) -> Policy:
        return utl.make_eps_greedy_policy(self.q, epsilon=self.epsilon)

    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        raise NotImplementedError

class Sarsa(Agent):
    def __init__(self, epsilon=0.05, gamma=0.99, alpha=0.1):
        self.q_values = {}
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy

    def policy(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 9)  # Explore: Random action
        else:
            q_values = [self.q_values.get((*state, a), 0) for a in range(9)]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)  # Exploit: Best action

    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        if (prev_state, prev_action) not in self.q_values:
            self.q_values[(prev_state, prev_action)] = 0
    
        if not done:
            next_action = self.policy(current_state)
            if (current_state, next_action) not in self.q_values:
                self.q_values[(current_state, next_action)] = 0
            
            q_update = prev_reward + self.gamma * self.q_values[(current_state, next_action)]
        else:
            q_update = prev_reward

        td_error = q_update - self.q_values[(prev_state, prev_action)]
        self.q_values[(prev_state, prev_action)] += self.alpha * td_error

        action = self.policy(current_state)

        return action




class QLearningAgent(Agent):
    def __init__(self):
        self.q_values = {}
        self.gamma = 0.99
        self.alpha = 0.1
        self.nA = 9
        
    def agent_step(self, prev_state: State, prev_action: Action, reward: float, current_state: State, done: bool) -> Action:
        if (prev_state, prev_action) not in self.q_values:
            self.q_values[(prev_state, prev_action)] = 0.0  # Ensure it's a float

        if done:
            q_update = reward  # No future reward if done
        else:
            max_q = max([self.q_values.get((current_state, a), 0) for a in range(self.nA)])
            q_update = reward + self.gamma * max_q  # Q-learning update rule

        self.q_values[(prev_state, prev_action)] += self.alpha * (q_update - self.q_values[(prev_state, prev_action)])

        if np.random.rand() < self.epsilon:
            next_action = np.random.randint(0, self.nA)  # Random action
        else:
            q_values = [self.q_values.get((current_state, a), 0) for a in range(self.nA)]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            next_action = np.random.choice(best_actions)  # Choose one of the best actions

        return next_action




def train_episode(agent: Agent, env: RaceTrack) -> tuple[list[State], list[Action], list[float]]:
    states = []
    rewards = []
    actions = []

    state = env.reset()  
    done = False
    truncated = False
    prev_state = state
    prev_action = None
    prev_reward = 0.0  

    while not (done or truncated):
        states.append(state)

        if prev_action is None:  
            action = np.random.randint(0, env.nA)  
        else:
            action = agent.agent_step(prev_state, prev_action, prev_reward, state, done)

        actions.append(action)
        next_state, reward, done, truncated = env.step(action)  

        rewards.append(float(reward))  

        if truncated:
            break  # Exit loop when truncated

        prev_state = state
        prev_action = action
        prev_reward = reward
        state = next_state

    return states, actions, rewards







def td_control(env: RaceTrack, agent_class: type[Agent], info: dict[str, Any], num_episodes: int) -> tuple[list[float, Agent]]:
    agent = agent_class()
    agent.agent_init(info)

    # Set seed
    seed = info['seed']
    np.random.seed(seed)
    random.seed(seed)

    all_returns = []

    for j in range(num_episodes):
        states, actions, rewards = train_episode(agent, env)
        ep_ret = np.sum(rewards)
        if j % 10 == 0:
            print(f"Episode {j}: sum of rewards = {ep_ret}, initial state: {states[0]}, last state: {states[-1]}")
        all_returns.append(ep_ret)

    return all_returns, agent