import gymnasium as gym, numpy as np, matplotlib.pyplot as plt

# Set the matplotlib backend
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend which works well across different platforms

# Constants
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODE = 25000
SHOW_EVERY = 3000

# Initialize environment
env = gym.make("MountainCar-v0", render_mode="human" if SHOW_EVERY > 0 else None)
ep_rewards = []  # List to store rewards for each episode
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODE//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# print(q_table)
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODE):
    episode_reward = 0
    state, _ = env.reset()  # Properly unpack state from reset()
    discrete_state = get_discrete_state(state)
    done = False

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}")

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)  # Fixed typo in randint

        new_state, reward, terminated, truncated, _ = env.step(action)  # Updated for new Gymnasium API
        done = terminated or truncated
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.unwrapped.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Add episode reward to the list
    ep_rewards.append(episode_reward)
    
    # Update aggregated rewards every SHOW_EVERY episodes
    if episode > 0 and episode % SHOW_EVERY == 0:
        latest_rewards = ep_rewards[-SHOW_EVERY:]
        average_reward = sum(latest_rewards) / len(latest_rewards)
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(latest_rewards))
        aggr_ep_rewards['min'].append(min(latest_rewards))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

    # Decay epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    np.save(f'qtables/{episode}-qtable.npy',q_table)
env.close()    

# Create a new figure with specified size
plt.figure(figsize=(12, 8))

# Plot the rewards
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Average Rewards", color='blue')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max Rewards", color='green')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min Rewards", color='red')

# Add labels and title
plt.title('Mountain Car Training Progress', fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='lower right', fontsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()

# Save the plot (optional)
plt.savefig('training_progress.png')