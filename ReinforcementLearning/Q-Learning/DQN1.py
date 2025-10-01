import gymnasium as gym, numpy as np
import random
from collections import deque
import torch, torch.nn as nn, torch.optim as optim 


# NN for Q-vals
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

# Hyperparameters

ENV_NAME = "CartPole-v1"
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEM_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10

# env setup

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr = LR)
memory = deque(maxlen=MEM_SIZE)

epsilon = EPS_START


# helper func
def  select_action(state, epsilon):
    if random.random() < epsilon:
        return random. randrange(action_dim)
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            return policy_net(state_t).argmax().item()

def train_step():
    if len(memory) < BATCH_SIZE: return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Q(s,a)
    q_values = policy_net(states).gather(1, actions).squeeze()

    # maxQ(s', a')
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q = rewards + (1-dones) * GAMMA * max_next_q

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Training loop

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):  # max steps per episode
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        train_step()

        if done:
            break

    # Epsilon decay
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

watch_episodes = 5
env = gym.make(ENV_NAME, render_mode="human")

for ep in range(watch_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = select_action(state, epsilon=0.0)  # always greedy (best action)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    print(f"Watch Episode {ep+1}: Reward = {total_reward}")
env.close() 