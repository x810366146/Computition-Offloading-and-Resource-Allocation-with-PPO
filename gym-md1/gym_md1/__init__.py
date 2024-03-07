from gymnasium.envs.registration import register

register(
    id="gym_md1/MD1Model-v0",
    entry_point="gym_md1.envs:MD1ModelEnv",
    max_episode_steps=300,
)