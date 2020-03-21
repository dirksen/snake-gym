import init_seed

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import math
from collections import deque
from keras.utils import to_categorical


class SnakeAction(object):
    WEST = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3


class BoardColor(object):
    BODY_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    SPACE_COLOR = np.array([255,255,255], dtype=np.uint8)


class MaurockSnakeEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 50
    }

    def __init__(self):
        self.score = 0
        self.width = 10
        self.height = 10

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=0, high=2, shape=(self.width, self.height, 1), dtype=np.uint8)

        self.snake = Snake()
        self.foods = []
        self.n_foods = 1
        self.viewer = None
        self.np_random = np.random

    def set_foods(self, n):
        self.n_foods = n

    def reset(self):
        self.score = 0
        self.snake.body.clear()
        self.foods.clear()
        empty_cells = self.get_empty_cells()
        empty_cells = self.snake.init(empty_cells, self.np_random)
        self.foods = [empty_cells[i] for i in self.np_random.choice(len(empty_cells), self.n_foods, replace=False)]
        # self.foods = [( self.width/2, self.height/2 - 2 )]
        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        snake_tail = self.snake.step(action)

        self.snake.reward = 0.

        if self.snake.head in self.foods:
            self.score += 1
            self.snake.reward += 1
            self.snake.body.append(snake_tail)
            self.foods.remove(self.snake.head)
            empty_cells = self.get_empty_cells()
            food = empty_cells[self.np_random.choice(len(empty_cells))]
            self.foods.append(food)

        #snake collided wall
        if self.is_collided_wall(self.snake.head):
            self.snake.reward -= 1
            self.snake.done = True

        #snake bite itself
        if self.snake.head in list(self.snake.body)[1:]:
            self.snake.reward -= 1
            self.snake.done = True

        # self.snake.reward = np.clip(self.snake.reward, -1., 1.)

        return self.get_observation(), self.snake.reward, self.snake.done, {}

    def is_dangerous(self, x, y):
        pos = ( x, y )
        return self.is_collided_wall(pos) or pos in list(self.snake.body)[2:]

    def get_observation(self):
        x, y = self.snake.head
        heading = self.snake.curr_act
        immediate_dangers = np.zeros(4)
        if heading:
            immediate_dangers = deque([
                self.is_dangerous(x - 1, y),
                self.is_dangerous(x + 1, y),
                self.is_dangerous(x, y - 1),
                self.is_dangerous(x, y + 1),
            ])
            # immediate_dangers.rotate(heading)
        heading = to_categorical(heading, 4) if heading is not None else np.zeros((4,))
        fx, fy = self.foods[0]
        food_orientation = [
            fx < x,
            fy < y,
            fx > x,
            fy > y
        ]
        observation = np.concatenate(( immediate_dangers, heading, food_orientation ))
        # convert booleans to integers
        return np.multiply(observation, 1).reshape(1,-1)

    def get_image(self):
        board_width = 20 * self.width
        board_height = 20 * self.height
        cell_size = 20

        board = Board(board_height, board_width)
        for x, y in self.snake.body:
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.BODY_COLOR)

        for food in self.foods:
            x, y = food
            board.fill_cell((x*cell_size, y*cell_size), cell_size, BoardColor.FOOD_COLOR)
        return board.board

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        for cell in self.snake.body:
            if cell in empty_cells:
                empty_cells.remove(cell)
        for food in self.foods:
            if food in empty_cells:
                empty_cells.remove(food)
        return empty_cells

    def is_collided_wall(self, head):
        x, y = head
        if x < 0 or x > (self.width - 1) or y < 0 or y > (self.height - 1):
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

    def __init__(self):
        self.body = deque()
        self.curr_act = None
        self.prev_act = None
        self.done = False
        self.reward = 0.

    def step(self, action):
        if not self.done:
            if not self.is_valid_action(action):
                action = self.curr_act
            self.curr_act = action
            x, y = self.head
            if action == SnakeAction.WEST:
                self.body.appendleft((x, y - 1))
            if action == SnakeAction.EAST:
                self.body.appendleft((x, y + 1))
            if action == SnakeAction.NORTH:
                self.body.appendleft((x - 1, y))
            if action == SnakeAction.SOUTH:
                self.body.appendleft((x + 1, y))
            return self.body.pop()

    @property
    def head(self):
        return self.body[0]

    def is_valid_action(self, action):
        if len(self.body) == 1:
            return True

        horizontal_actions = [SnakeAction.WEST, SnakeAction.EAST]
        vertical_actions = [SnakeAction.NORTH, SnakeAction.SOUTH]

        if self.curr_act in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions

    def init(self, empty_cells, np_random):
        self.body.clear()
        self.done = False
        self.reward = 0.
        self.curr_act = None
        self.prev_act = None
        start_head = empty_cells[np_random.choice(len(empty_cells))]
        # start_head = empty_cells[int( len(empty_cells) / 2 ) + 2]
        self.body.appendleft( start_head )
        empty_cells.remove(start_head)
        return empty_cells


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = BoardColor.SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x+cell_size, y:y+cell_size, :] = color
