import numpy as np
from environment import RaceTrack

# Load the last learned Q-values from SARSA and Q-learning
with open('data/td_sarsa_agents.npy', 'rb') as f:
    sarsa_agents = np.load(f, allow_pickle=True)

with open('data/td_qlearning_agents.npy', 'rb') as f:
    qlearning_agents = np.load(f, allow_pickle=True)
    
sarsa_last = sarsa_agents[-1]
qlearning_last = qlearning_agents[-1]    

env = RaceTrack(track_map='c', render_mode=None)

epsilon = 0.2  # Epsilon for ε-greedy policy

def evaluate_agent(agent, epsilon, episodes=5):
    accumulated_returns = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_return = 0

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:  # Exploration
                action = np.random.choice(env.nA)
            else:  # Exploitation
                if state in agent.q_values:
                    action = max(agent.q_values[state], key=agent.q_values[state].get)
                else:
                    action = np.random.choice(env.nA)  # Random action if state is unseen

            next_state, reward, done, _ = env.step(action)
            total_return += reward
            state = next_state

        accumulated_returns.append(total_return)

    return accumulated_returns

# Evaluate both agents
sarsa_returns = evaluate_agent(sarsa_last, epsilon)
qlearning_returns = evaluate_agent(qlearning_last, epsilon)

# Report returns
print("SARSA Returns:")
for i, ret in enumerate(sarsa_returns):
    print(f'Episode {i+1}: Accumulated Return = {ret}')

print("\nQ-Learning Returns:")
for i, ret in enumerate(qlearning_returns):
    print(f'Episode {i+1}: Accumulated Return = {ret}')

# Compare the sensitivity of both algorithms
average_sarsa = np.mean(sarsa_returns)
average_qlearning = np.mean(qlearning_returns)

print(f"\nAverage SARSA Return: {average_sarsa}")
print(f"Average Q-Learning Return: {average_qlearning}")
