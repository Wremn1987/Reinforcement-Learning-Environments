import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

class QLearningAgent:
    """
    A simple Q-Learning agent for discrete state and action spaces.
    This agent learns an optimal policy by updating a Q-table based on
    the rewards received from the environment.
    """
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """
        Initializes the Q-Learning agent.
        
        Args:
            num_states (int): The number of possible states in the environment.
            num_actions (int): The number of possible actions the agent can take.
            learning_rate (float): The learning rate (alpha) for updating Q-values.
            discount_factor (float): The discount factor (gamma) for future rewards.
            exploration_rate (float): The initial exploration rate (epsilon) for epsilon-greedy policy.
            exploration_decay (float): The decay rate for the exploration rate.
            min_exploration_rate (float): The minimum exploration rate.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))
        logger.info(f"Initialized QLearningAgent with {num_states} states and {num_actions} actions.")

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.
        
        Args:
            state (int): The current state.
            
        Returns:
            int: The chosen action.
        """
        # Exploration: choose a random action
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, self.num_actions - 1)
            logger.debug(f"State {state}: Exploring - chose random action {action}")
            return action
        
        # Exploitation: choose the action with the highest Q-value for the current state
        # If multiple actions have the same max Q-value, choose randomly among them
        q_values = self.q_table[state, :]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        action = random.choice(best_actions)
        logger.debug(f"State {state}: Exploiting - chose best action {action} with Q-value {max_q}")
        return action

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Updates the Q-value for a given state-action pair using the Q-learning update rule.
        
        Args:
            state (int): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (int): The resulting state.
            done (bool): Whether the episode has ended.
        """
        # Get the maximum Q-value for the next state
        if done:
            max_next_q = 0.0 # No future rewards if the episode is done
        else:
            max_next_q = np.max(self.q_table[next_state, :])
            
        # Calculate the new Q-value
        current_q = self.q_table[state, action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update the Q-table
        self.q_table[state, action] = new_q
        logger.debug(f"Updated Q-value for state {state}, action {action}: {current_q:.4f} -> {new_q:.4f}")

    def decay_exploration_rate(self):
        """
        Decays the exploration rate according to the decay factor.
        """
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
        logger.debug(f"Exploration rate decayed to {self.exploration_rate:.4f}")

    def get_optimal_policy(self):
        """
        Returns the optimal policy derived from the learned Q-table.
        
        Returns:
            np.ndarray: An array where the i-th element is the optimal action for state i.
        """
        return np.argmax(self.q_table, axis=1)

if __name__ == "__main__":
    # Example usage: A simple grid world environment
    logger.info("Running example for QLearningAgent in a simple environment...")
    
    # Define a simple environment: 5 states, 2 actions (left, right)
    # Goal is state 4, starting at state 0
    num_states = 5
    num_actions = 2
    
    agent = QLearningAgent(num_states, num_actions)
    
    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        state = 0 # Start state
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            
            # Simulate environment step
            if action == 1: # Move right
                next_state = min(state + 1, num_states - 1)
            else: # Move left
                next_state = max(state - 1, 0)
                
            # Reward logic
            if next_state == num_states - 1:
                reward = 10 # Reached goal
                done = True
            else:
                reward = -1 # Step penalty
                
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.decay_exploration_rate()
        if (episode + 1) % 20 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes} completed. Total Reward: {total_reward}")
            
    # Print learned Q-table and optimal policy
    logger.info("Training complete.")
    logger.info(f"Learned Q-table:\n{agent.q_table}")
    logger.info(f"Optimal Policy: {agent.get_optimal_policy()}")
