import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture.
    A simple feedforward neural network that takes the state as input
    and outputs Q-values for each action.
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    A Deep Q-Learning agent that uses a neural network to approximate the Q-function.
    It employs experience replay and a target network for stable learning.
    """
    def __init__(self, state_size, action_size, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"DQNAgent initialized on device: {self.device}")

        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size).to(self.device)
        self.qnetwork_target = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=int(1e5))  # D_SIZE
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.tau = 1e-3  # For soft update of target parameters
        self.lr = 5e-4 # Learning rate
        self.update_every = 4 # How often to update the network

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self._sample_experiences()
                self._learn(experiences, self.gamma)

    def choose_action(self, state, eps=0.0):
        """
        Returns actions for given state as per epsilon-greedy policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _sample_experiences(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def _learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

if __name__ == "__main__":
    logger.info("Running example for DQNAgent...")
    
    # Define a dummy environment for testing
    state_size = 4 # e.g., cartpole state
    action_size = 2 # e.g., cartpole actions
    
    agent = DQNAgent(state_size, action_size)
    
    # Simulate some steps
    state = np.random.rand(state_size)
    for i in range(100):
        action = agent.choose_action(state, eps=0.1)
        next_state = np.random.rand(state_size)
        reward = random.choice([-1, 0, 1])
        done = random.choice([True, False])
        
        agent.step(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            state = np.random.rand(state_size)
            
    logger.info(f"Memory size: {len(agent.memory)}")
    logger.info("DQNAgent example completed.")
