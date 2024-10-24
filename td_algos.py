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
    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        action = 0
        
        if done:
            next_q_value = 0.0
            action = None
        else:
            action = self.get_current_policy()(current_state)
            current_state_action = (*current_state, action)

            if current_state_action not in self.q:
                self.q[current_state_action] = 0.0
                
            next_q_value = self.q.get(current_state_action, 0.0)
            
        prev_state_action = (*prev_state, prev_action)    
        
        if prev_state_action not in self.q:
            self.q[prev_state_action] = 0.0    

        prev_q_val = self.q.get(prev_state_action, 0.0)
        
        self.q[prev_state_action] = prev_q_val + self.step_size * (prev_reward + self.discount * next_q_value - prev_q_val)

        return action



class QLearningAgent(Agent):
    def __init__(self):
        super().__init__()
        
    def agent_step(self, prev_state: State, prev_action: Action, prev_reward: float, current_state: State, done: bool) -> Action:
        action = 0 

        if done:
            max_q = 0.0  
        else:            
            max_q = max([self.q.get((*current_state, a), 0.0) for a in range(9)])

        prev_state_action = (*prev_state, prev_action)
        prev_q_val = self.q.get(prev_state_action, 0.0)
        
        self.q[prev_state_action] = prev_q_val + self.step_size * (prev_reward + self.discount * max_q - prev_q_val)

        action = self.get_current_policy()(current_state)

        return action




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

    action = agent.get_current_policy()(state)
    
    while not (done or truncated):
        next_state, reward, done, truncated = env.step(action)  
        states.append(state)

        actions.append(action)

        rewards.append(float(reward))  
        
        next_action = agent.agent_step(state, action, reward, next_state, done)

        state = next_state
        action = next_action

    return states, actions, rewards



def td_control(env: RaceTrack, agent_class: type[Agent], info: dict[str, Any], num_episodes: int) -> tuple[list[float, Agent]]:
    agent = agent_class()
    agent.agent_init(info)

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