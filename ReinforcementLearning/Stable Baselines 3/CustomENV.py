import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


class SnekEnv(gym.Env):

    def __init__(self):
        super(SnekEnv, self).__init__()
        # 4 discrete actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        # Observation space (snake head, apple delta, length, history of actions)
        self.observation_space = spaces.Box(
            low=-500, high=500,
            shape=(5 + SNAKE_LEN_GOAL,),
            dtype=np.float32
        )

    def step(self, action):
        self.prev_actions.append(action)
        cv2.imshow('Snake', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500, 500, 3), dtype='uint8')

        # Draw Apple
        cv2.rectangle(self.img,
                      (self.apple_position[0], self.apple_position[1]),
                      (self.apple_position[0] + 10, self.apple_position[1] + 10),
                      (0, 0, 255), 3)

        # Draw Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,
                          (position[0], position[1]),
                          (position[0] + 10, position[1] + 10),
                          (0, 255, 0), 3)

        # Move head
        if action == 1:
            self.snake_head[0] += 10
        elif action == 0:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10

        # Eat apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0, list(self.snake_head))
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # Check collisions
        if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(self.img, f'Your Score is {self.score}', (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Snake', self.img)
            self.done = True

        self.total_reward = len(self.snake_position) - 3  # default length = 3
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.done:
            self.reward = -10

        # Build observation
        head_x, head_y = self.snake_head
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float32)

        # Gymnasium API → 5 return values
        terminated = self.done
        truncated = False  # optional: add max steps limit

        return observation, self.reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.img = np.zeros((500, 500, 3), dtype='uint8')
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.snake_head = [250, 250]

        self.prev_reward = 0
        self.done = False

        head_x, head_y = self.snake_head
        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        self.prev_actions = deque(maxlen=SNAKE_LEN_GOAL)
        for _ in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1)

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation, dtype=np.float32)

        return observation, {}
