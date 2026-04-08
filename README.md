# Reinforcement Learning Environments

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green?style=flat-square&logo=gymnasium)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red?style=flat-square&logo=pytorch)](https://pytorch.org/)

This repository serves as a playground for exploring and implementing various reinforcement learning (RL) algorithms. It includes classic control problems from Gymnasium (formerly OpenAI Gym) and custom-built environments to test and understand different RL paradigms.

## Table of Contents
- [Introduction](#introduction)
- [Algorithms Implemented](#algorithms-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](#environments)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Reinforcement Learning is a powerful paradigm where an agent learns to make decisions by interacting with an environment to maximize a cumulative reward. This project provides hands-on examples of popular RL algorithms and their application.

## Algorithms Implemented
- **Q-Learning:** Model-free, off-policy RL algorithm.
- **SARSA:** Model-free, on-policy RL algorithm.
- **Deep Q-Networks (DQN):** Combines Q-Learning with deep neural networks.
- **Policy Gradients (REINFORCE):** Basic policy-based method.

## Project Structure
```
.gitignore
README.md
requirements.txt
src/
├── __init__.py
├── q_learning_agent.py
└── dqn_agent.py
environments/
├── custom_env.py
└── __init__.py
notebooks/
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Wremn1987/Reinforcement-Learning-Environments.git
   cd Reinforcement-Learning-Environments
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- To run the Q-Learning agent on CartPole:
  ```bash
  python src/q_learning_agent.py --env CartPole-v1
  ```
- To run the DQN agent on LunarLander:
  ```bash
  python src/dqn_agent.py --env LunarLander-v2
  ```

## Environments
- **Gymnasium Classic Control:** CartPole-v1, MountainCar-v0, etc.
- **Custom Environments:** `environments/custom_env.py` provides a template for creating your own RL environments.

## Contributing
New algorithm implementations or custom environments are highly appreciated.

## License
This project is licensed under the MIT License.
