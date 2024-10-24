import random
import matplotlib.pyplot as plt
import numpy as np
import utils as utl
from environment import RaceTrack
from td_algos import Sarsa, td_control

# Set initial seed
initial_seed = 0
render_mode = None
num_episodes = 2000

# Create the environment
env = RaceTrack(track_map='c', render_mode=render_mode)

# Prepare lists to store results
returns = []
agents = []

# Run the loop with a different seed for each run
for i in range(5):
    # Update the seed
    seed = initial_seed + i
    np.random.seed(seed)
    random.seed(seed)

    # Update the info dictionary with the new seed
    info = {
        "env": env,
        "step_size": 0.1,
        "epsilon": 0.05,
        "discount": 0.99,
        "seed": seed
    }

    # Perform TD control using the Sarsa agent
    all_returns, agent = td_control(env, agent_class=Sarsa, info=info, num_episodes=num_episodes)

    # Store the results
    returns.append(all_returns)
    agents.append(agent)

# Save the results to files
with open('data/td_sarsa_returns.npy', 'wb') as f:
    np.save(f, returns)

with open('data/td_sarsa_agents.npy', 'wb') as g:
    np.save(g, agents)

# Plot the results
plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(returns)
