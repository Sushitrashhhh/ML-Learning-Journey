import numpy as np
import random

GRID_SIZE = 5
ACTIONS = ['up','down','right','left']

GOAL=(4,4)
TRAP=(3,3)

def step(state, action):
    x,y = state
    if action == 'up': x =max(0,x-1)
    elif action == 'down': x = min(GRID_SIZE-1, x+1)
    elif action == 'left': y = max(0, y-1)
    elif action == 'right': y = min(GRID_SIZE-1, y+1)
    new_state = (x, y)

    if new_state == GOAL:
        return new_state, 10, True
    elif new_state == TRAP:
        return new_state, -10, True
    else:
        return new_state, -1, False
    
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
epsilon = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01  


for ep in range(EPISODES):
    state = (0, 0)
    done = False

    while not done:
        # Exploration vs exploitation
        if random.uniform(0,1) < epsilon:
            action_idx = random.randint(0, len(ACTIONS)-1)
        else:
            action_idx = np.argmax(q_table[state[0], state[1]])

        action = ACTIONS[action_idx]
        new_state, reward, done = step(state, action)

        # Q-learning update
        old_q = q_table[state[0], state[1], action_idx]
        max_future_q = np.max(q_table[new_state[0], new_state[1]])
        new_q = (1-LEARNING_RATE)*old_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
        q_table[state[0], state[1], action_idx] = new_q

        state = new_state

    # Decay epsilon
    epsilon = max(MIN_EPSILON, epsilon*EPSILON_DECAY)

print("Training finished ✅")