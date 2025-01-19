import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN as SB3_DQN  # Correct import
import numpy as np
import random
from collections import deque
from gym.wrappers.record_video import RecordVideo

# Define the neural network architecture with 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  
            nn.ReLU(),                 
            nn.Linear(128, output_dim) 
        )

    def forward(self, x):
        return self.fc(x)

num_episodes = 500
max_steps = 200
gamma = 0.99 
epsilon = 1.0  
epsilon_min = 0.01 
epsilon_decay = 0.995  
batch_size = 64  
memory_size = 10000  

env = gym.make("CartPole-v1")


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(state_dim, action_dim)
model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

state, _ = env.reset()  
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
q_values = model(state)
action = torch.argmax(q_values).item()


replay_buffer = deque(maxlen=memory_size)

# Function to select an action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:  # Explore
        return env.action_space.sample()
    else:  # Exploit
        q_values = model(state)
        return torch.argmax(q_values).item()

# Main training loop
for episode in range(num_episodes):
    state, _ = env.reset()  # Corrected to match the latest Gym API
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    
    for t in range(max_steps):
        # Select an action
        action = select_action(state, epsilon)

        # Take the action in the environment
        next_state, reward, done, _, _ = env.step(action)  # Corrected to match the latest Gym API
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # Store experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Update state and total reward
        state = next_state
        total_reward += reward

        # If the episode is done, break
        if done:
            break

        # Train the model if replay buffer has enough samples
        if len(replay_buffer) >= batch_size:
            # Sample a batch of experiences
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = torch.cat(states)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # Compute Q-values and target Q-values
            q_values = model(states).gather(1, actions)
            next_q_values = model(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))

            # Compute loss
            loss = loss_fn(q_values, target_q_values)

            # Perform gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print progress
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")



