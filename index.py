import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN
import numpy as np


# Define the neural network architecture with 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Fully connected layer: input -> 128 neurons
            nn.ReLU(),                  # Activation function: ReLU
            nn.Linear(128, output_dim)  # Fully connected layer: 128 -> output_dim neurons
        )

    def forward(self, x):
        return self.fc(x)


env = gym.make("CartPole-v1")


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(state_dim, action_dim)
model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

state, _ = env.reset()
print("State shape:", state.shape) 
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
print("State tensor shape:", state.shape)
q_values = model(state)
print("Q-values:", q_values)
action = torch.argmax(q_values).item()
print("Selected Action:", action)