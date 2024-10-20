import numpy as np
from environment import RaceTrack
import algorithms as algo

# Load the last learned state-action values from previous runs
with open('data/fv_mc_sa_values.npy', 'rb') as f:
    all_sa_values = np.load(f, allow_pickle=True)
    
# Get the last learned state-action value dictionary
last_sa_values = all_sa_values[-1]

# Set the environment
env = RaceTrack(track_map='c', render_mode=None)

# Set epsilon to 0 for greedy policy
epsilon = 0.5

# Evaluate the greedy policy on 5 episodes
num_episodes = 5
accumulated_returns = []

for episode in range(num_episodes):
    total_return = 0
    state = env.reset()
    done = False

    while not done:
        # Choose the greedy action (argmax of the state-action values)
        if state in last_sa_values:
            action = max(last_sa_values[state], key=last_sa_values[state].get)
        else:
            action = np.random.choice(env.nA)
        
        next_state, reward, done, _ = env.step(action)
        total_return += reward
        state = next_state

    accumulated_returns.append(total_return)

# Report the accumulated returns for each episode
for i, ret in enumerate(accumulated_returns):
    print(f'Episode {i+1}: Accumulated Return = {ret}')
