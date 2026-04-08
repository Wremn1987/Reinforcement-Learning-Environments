from gymnasium.envs.registration import register

register(
    id='CustomGridWorld-v0',
    entry_point='environments.custom_env:CustomGridWorldEnv',
    max_episode_steps=300,
    reward_threshold=100.0,
)
