import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = gym.make("LunarLander-v3", render_mode="human")

# Create PPO agent with GPU
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda"   # 👈 Forces training on GPU if available
)

# Train agent
model.learn(total_timesteps=200_000)

# Save the trained model
model.save("ppo_lunarlander")

# Test the trained agent
env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
