# src/q_learning_agent.py

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningAgent:
    """
    A Q-Learning agent implementation for discrete action spaces.
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay_rate=0.001, min_epsilon=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        # Assuming discrete observation space for simplicity (e.g., CartPole-v1 state discretization)
        # For continuous spaces, a function approximator (like a neural network) would be needed.
        if isinstance(env.observation_space, gym.spaces.Box):
            # Discretize continuous observation space for Q-table
            self.q_table = self._create_q_table_for_continuous_env(env)
        else:
            self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def _create_q_table_for_continuous_env(self, env):
        # Example discretization for CartPole-v1
        # Cart Position: -2.4 to 2.4 -> 10 bins
        # Cart Velocity: -4 to 4 -> 10 bins
        # Pole Angle: -0.2095 to 0.2095 -> 10 bins
        # Pole Angular Velocity: -4 to 4 -> 10 bins
        self.pos_space = np.linspace(-2.4, 2.4, 10)
        self.vel_space = np.linspace(-4, 4, 10)
        self.ang_space = np.linspace(-0.2095, 0.2095, 10)
        self.ang_vel_space = np.linspace(-4, 4, 10)
        return np.zeros((len(self.pos_space)+1, len(self.vel_space)+1, len(self.ang_space)+1, len(self.ang_vel_space)+1, env.action_space.n))

    def _get_state_index(self, state):
        if isinstance(self.env.observation_space, gym.spaces.Box):
            # Discretize state for continuous environments
            pos_idx = np.digitize(state[0], self.pos_space)
            vel_idx = np.digitize(state[1], self.vel_space)
            ang_idx = np.digitize(state[2], self.ang_space)
            ang_vel_idx = np.digitize(state[3], self.ang_vel_space)
            return (pos_idx, vel_idx, ang_idx, ang_vel_idx)
        else:
            return state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore action space
        else:
            state_idx = self._get_state_index(state)
            return np.argmax(self.q_table[state_idx]) # Exploit learned values

    def update_q_table(self, state, action, reward, next_state, done):
        state_idx = self._get_state_index(state)
        next_state_idx = self._get_state_index(next_state)

        old_value = self.q_table[state_idx + (action,)]
        next_max = np.max(self.q_table[next_state_idx])

        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max * (1 - done) - old_value)
        self.q_table[state_idx + (action,)] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

def train_q_learning(env_name='CartPole-v1', num_episodes=1000):
    env = gym.make(env_name)
    agent = QLearningAgent(env)

    rewards_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done or truncated)
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
    plt.title(f'Q-Learning Training on {env_name}' )
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(f'q_learning_{env_name}_rewards.png')
    plt.show()

if __name__ == '__main__':
    train_q_learning(env_name='CartPole-v1', num_episodes=500) # Reduced episodes for faster execution
