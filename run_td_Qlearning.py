import random
import matplotlib.pyplot as plt
import numpy as np
import utils as utl
from environment import RaceTrack
from td_algos import QLearningAgent, td_control

# Set base seed
initial_seed = 0
render_mode = None
num_episodes = 2000

env = RaceTrack(track_map='c', render_mode=render_mode)

returns = []
agents = []
 
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
    all_returns, agent = td_control(env, agent_class=QLearningAgent, info=info, num_episodes=num_episodes)

    # Store the results
    returns.append(all_returns)
    agents.append(agent)
    
# Save the results
with open('data/td_qlearning_returns.npy', 'wb') as f:
    np.save(f, returns)
with open('data/td_qlearning_agents.npy', 'wb') as g:
    np.save(g, agents)

# Plot the returns
plt.figure(figsize=(15, 7))
plt.grid()
utl.plot_many(returns)
