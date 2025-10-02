from stable_baselines3.common.env_checker import check_env
from CustomENV import SnekEnv
import os, time
from stable_baselines3 import PPO

# Directories for saving
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Create env
env = SnekEnv()

# ✅ Check environment validity
check_env(env)

# Create PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device="cuda")

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
