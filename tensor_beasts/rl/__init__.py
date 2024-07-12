from gymnasium.envs.registration import register

register(
    id='tensor-beasts-v0',
    entry_point='tensor_beasts.rl.envs:TensorBeastsEnv'
)
