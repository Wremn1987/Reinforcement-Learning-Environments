# environments/custom_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomGridWorldEnv(gym.Env):
    """
    A simple custom GridWorld environment for reinforcement learning.
    The agent navigates a grid to reach a goal, avoiding obstacles.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observation space: agent's position (x, y)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=int)

        # Action space: 0: right, 1: up, 2: left, 3: down
        self.action_space = spaces.Discrete(4)

        # Define the grid layout (0: empty, 1: obstacle, 2: goal, 3: start)
        # Example 5x5 grid
        self._grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 2]  # Goal at (4, 4)
        ])
        self._start_pos = (0, 0)
        self._goal_pos = (4, 4)
        self._obstacle_positions = [(1,1), (1,3), (3,1), (3,3)]

        self._action_to_direction = {
            0: np.array([1, 0]),  # Right
            1: np.array([0, 1]),  # Up
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1])  # Down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._goal_pos, ord=1)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array(self._start_pos)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        new_location = self._agent_location + direction

        # Clip the new location to stay within grid boundaries
        new_location = np.clip(new_location, 0, self.size - 1)

        # Check for collision with obstacles
        if tuple(new_location) in self._obstacle_positions:
            reward = -10.0 # Penalty for hitting an obstacle
            terminated = False # Does not terminate, just penalizes
            self._agent_location = self._agent_location # Agent stays in place
        else:
            self._agent_location = new_location
            reward = -1.0 # Penalty for each step
            terminated = bool(np.array_equal(self._agent_location, self._goal_pos))
            if terminated:
                reward = 100.0 # Reward for reaching the goal

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        pix_square_size = self.window_size / self.size  # The size of a single grid square in pixels

        # Draw the grid lines
        for x in range(self.size + 1):
            pygame.draw.line(canvas,
                             0, # Black
                             (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size),
                             width=1)
            pygame.draw.line(canvas,
                             0, # Black
                             (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x),
                             width=1)
        
        # Draw the goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0), # Green
            pygame.Rect(
                pix_square_size * self._goal_pos[0],
                pix_square_size * self._goal_pos[1],
                pix_square_size,
                pix_square_size,
            ),
        )

        # Draw obstacles
        for obs_pos in self._obstacle_positions:
            pygame.draw.rect(
                canvas,
                (255, 0, 0), # Red
                pygame.Rect(
                    pix_square_size * obs_pos[0],
                    pix_square_size * obs_pos[1],
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255), # Blue
            ((self._agent_location + 0.5) * pix_square_size),
            pix_square_size / 3,
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    env = CustomGridWorldEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0

    for _ in range(100): # Run for a maximum of 100 steps
        action = env.action_space.sample() # Take a random action
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if done or truncated:
            break
    env.close()
    print(f"Total reward: {total_reward}")
