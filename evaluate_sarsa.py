import numpy as np
from environment import RaceTrack

# Load the trained SARSA agents from the previous runs
# with open('data/td_sarsa_agents.npy', 'rb') as f:
with open('data/td_qlearning_agents.npy','rb') as f:   # Qlearning
    agents = np.load(f, allow_pickle=True)

env = RaceTrack(track_map='c', render_mode=None)

epsilon = 0  # Greedy policy

accumulated_returns = []

for i, agent in enumerate(agents):
    agent.epsilon = epsilon  # Set epsilon to 0 for greedy policy
    state = env.reset()
    done = False
    total_return = 0

    # Initialize action by selecting the greedy action
    if state in agent.q_values:
        action = max(agent.q_values[state], key=agent.q_values[state].get)
    else:
        action = np.random.choice(env.nA)
    
    while not done:
        next_state, reward, done, _ = env.step(action)
        total_return += reward

        if next_state in agent.q_values:
            next_action = max(agent.q_values[next_state], key=agent.q_values[next_state].get)
        else:
            next_action = np.random.choice(env.nA)

        # Update state and action for the next step
        state = next_state
        action = next_action

    accumulated_returns.append(total_return)

# Report accumulated returns for each episode
for i, ret in enumerate(accumulated_returns):
    print(f'Episode {i+1}: Accumulated Return = {ret}')
