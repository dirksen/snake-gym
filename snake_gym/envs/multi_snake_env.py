
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math
from collections import deque


class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)

SNAKE_COLOR = [np.array([255, 0, 0], dtype=np.uint8), \
               np.array([0, 0, 255], dtype=np.uint8), \
               np.array([255, 255, 0], dtype=np.uint8), \
               np.array([0, 255, 255], dtype=np.uint8), \
               np.array([255, 0, 255], dtype=np.uint8)]


class MultiSnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 50
    }

    def __init__(self, n_snakes):
        self.width = 20
        self.hight = 20

        self.action_space = spaces.Box(low=0, high=3, shape=(n_snakes, ), dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(400, 400, 3), dtype=np.uint8)

        self.n_snakes = n_snakes
        self.snakes = [Snake(i) for i in range(n_snakes)]
        self.foods = []
        self.viewer = None
        self.np_random = np.random

    def reset(self):
        empty_cells = self.get_empty_cells()
        for i in range(self.n_snakes):
            empty_cells = self.snakes[i].reset(empty_cells, self.np_random)
        self.foods = [empty_cells[i] for i in self.np_random.choice(len(empty_cells), 3)]
        return self.get_image()

    def snake_rebirth(self):
        snake = Snake()
        empty_cells = self.get_empty_cells()
        empty_cells = snake.init(empty_cells, self.np_random)
        return snake

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        for i in range(self.n_snakes):
            self.snakes[i].step(action[i])
        
        for snake in self.snakes:
            if snake.head in self.foods:
                snake.reward += 1.
                snake.grow()
                empty_cells = self.get_empty_cells()
                self.foods.remove(snake.head)
                if len(self.foods) < 10:
                    food = empty_cells[self.np_random.choice(len(empty_cells))]
                    self.foods.append(food) 

            if self.bite_others_or_itself(snake) or self.is_collided_wall(snake.head):
                snake.reward -= len(snake.body)
                snake.done = True
                self.foods.extend(list(snake.body)[1:])
                self.foods.append(snake.tail)
        
        rewards = []
        dones = []
        for snake in self.snakes:
            rewards.append(snake.reward)
            dones.append(snake.done)
            snake.reward = 0.
            if snake.done:
                empty_cells = self.get_empty_cells()
                snake.reset(empty_cells, self.np_random)
        
        return self.get_image(), rewards, dones, {}

    def bite_others_or_itself(self, this_snake):
        snakes = self.snakes.copy()
        other_snakes = snakes.remove(this_snake)
        all_body_cells = []
        for snake in snakes:
            all_body_cells.extend(list(snake.body))
        all_body_cells.extend(list(this_snake.body)[1:])
        return this_snake.head in all_body_cells


    def get_image(self):
        board_width = 400
        board_height = 400
        cell_size = int(board_width / self.width)

        board = Board(board_height, board_width)
        for snake in self.snakes:
            for x, y in snake.body:
                board.fill_cell((x*cell_size, y*cell_size), cell_size, snake.color)
        
        for food in self.foods:
            x, y = food
            board.fill_cell((x*cell_size, y*cell_size), cell_size, FOOD_COLOR)
        return board.board

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.hight)]
        for snake in self.snakes:
            for cell in snake.body:
                if cell in empty_cells:
                    empty_cells.remove(cell)
        for food in self.foods:
            if self.foods in empty_cells:
                empty_cells.remove(self.foods)
        return empty_cells

    def is_collided_wall(self, head):
        x, y = head
        if x < 0 or x > 19 or y < 0 or y > 19:
            return True
        return False

    def render(self, mode='human'):
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class Snake(object):

    def __init__(self, i):
        self.body = deque()
        self.color = SNAKE_COLOR[i]
        self.prev_action = None
        self.tail = None
        self.reward = 0.
        self.done = False
        
    def step(self, action):
        if not self.is_valid_action(action):
            action = self.prev_action
        self.prev_action = action
        x, y = self.head
        if action == SnakeAction.LEFT:
            self.body.appendleft((x, y - 1))
        if action == SnakeAction.RIGHT:
            self.body.appendleft((x, y + 1))
        if action == SnakeAction.UP:
            self.body.appendleft((x - 1, y))
        if action == SnakeAction.DOWN:
            self.body.appendleft((x + 1, y))
        self.tail = self.body.pop()
    
    def grow(self):
        self.body.append(self.tail)

    @property
    def head(self):
        return self.body[0]

    def is_valid_action(self, action):
        if len(self.body) == 1:
            return True
        
        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_action in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions
    
    def reset(self, empty_cells, np_random):
        self.reward = 0.
        self.done = False
        self.body.clear()
        start_head = empty_cells[np_random.choice(len(empty_cells))]
        self.body.appendleft(start_head)
        empty_cells.remove(start_head)
        return empty_cells


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color

