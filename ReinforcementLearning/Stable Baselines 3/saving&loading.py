import gymnasium as gym
from stable_baselines3 import PPO
import os
models_dir = "models/PPO"
logdir = 'logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('LunarLander-v3',render_mode="human")
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, device="cuda")
TIMESTEPS=5000

iters = 0
while True:
    iters+=1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name='test_run')
    model.save(f"{models_dir}/{TIMESTEPS*iters}")


# episodes = 10

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while  not done:
#         env.render()
#         obs, reward, done, info = env.step(env.action_space.sample())

# env.close()