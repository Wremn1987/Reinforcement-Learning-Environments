# src/dqn_agent.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class DQN(nn.Module):
    """Deep Q-Network (DQN) model."""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    A Deep Q-Network (DQN) agent implementation.
    """
    def __init__(self, env, state_size, action_size, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.005, min_epsilon=0.01, replay_buffer_size=10000, batch_size=64):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size

        self.memory = deque(maxlen=replay_buffer_size) # Replay buffer
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        # Compute Q-values for current states
        current_q_values = self.model(states).gather(1, actions)

        # Compute target Q-values
        next_q_values = self.model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # Compute loss and update model
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

def train_dqn(env_name='CartPole-v1', num_episodes=1000):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(env, state_size, action_size)

    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done or truncated)
            agent.learn()
            state = next_state
            episode_reward += reward
        
        agent.decay_epsilon()
        rewards_per_episode.append(episode_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Epsilon = {agent.epsilon:.2f}")

    env.close()
    print("
Training finished.")

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode)
    plt.title(f'DQN Training on {env_name}' )
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'dqn_{env_name}_rewards.png')
    plt.show()

if __name__ == '__main__':
    train_dqn(env_name='CartPole-v1', num_episodes=500) # Reduced episodes for faster execution
