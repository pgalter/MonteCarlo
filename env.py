import pygame
import numpy as np
from GameObject import GameObject, Agent
import gymnasium as gym
from enum import Enum
from Policy import Policy

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

    @staticmethod
    def items() -> list:
        return [e.value for e in Actions]

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "sim", "text"], "render_fps": 2}

    def __init__(self, render_mode=None, movement_penalty=0):
        """

        :param render_mode: ["human", "sim", "text"]
        :param movement_penalty: penalty for each move, to encourage shorter paths
        """
        super (CustomEnv, self).__init__()

        pygame.init()
        pygame.display.init()

        self.size = 8 # size of the square grid (8x8) in squares
        self.window_size = 512 # size of the (square) game window in pixels if render mode is "human"
        self.square_pix_size = self.window_size / self.size

        # initialize pygame tools
        self.window = None
        self.render_mode = render_mode
        self.clock = None

        # initialize game objects
        self._nemo = Agent('nemo', "nemo.png", self.square_pix_size, max_pos=self.size)
        self._anemone = GameObject('anemone', "anemone.png", self.square_pix_size)
        self._objects = {
            self._anemone: np.array((7,7), dtype=np.int32),
            GameObject('seaweed', "seaweed.png", self.square_pix_size): np.array((4,7), dtype=np.int32),
            GameObject('seaweed', "seaweed.png", self.square_pix_size): np.array((7,6), dtype=np.int32),
            GameObject('seaweed',"seaweed.png", self.square_pix_size): np.array((5,4), dtype=np.int32),
            GameObject('seaweed',"seaweed.png", self.square_pix_size): np.array((4,5), dtype=np.int32),
            GameObject('bag', "bag.png", self.square_pix_size): np.array((5,3), dtype=np.int32),
            GameObject('crab',"crab.png", self.square_pix_size): np.array((2,7), dtype=np.int32),
            GameObject('bottle', "bottle.png", self.square_pix_size): np.array((0,4), dtype=np.int32),
            GameObject('octopus', "octopus.png", self.square_pix_size): np.array((5,6), dtype=np.int32),
            GameObject('dory', "dory.png", self.square_pix_size): np.array((6,2), dtype=np.int32),
            GameObject('shark', "shark.png", self.square_pix_size, double=True): np.array((1,1), dtype=np.int32),
            GameObject('shark', "shark2.png", self.square_pix_size, double=True): np.array((3,2), dtype=np.int32),
        }

        # action and observation spaces
        self.action_space = gym.spaces.Discrete(4) # four possible actions, Left, Right, Up, Down
        self.observation_space = gym.spaces.Box(
                    low=0, high=np.array([self.size, self.size]), shape=(2,), dtype=np.int32
                )
        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }

        # rewards
        self._rewards = {'anemone': 10,
                        'bag': -1,
                        'crab': -5,
                        'bottle': -1,
                        'octopus': -5,
                        'dory': 0,
                        'shark': -10,
                        'move': -movement_penalty,
                        'seaweed': -2
                         }

        self.policy = None

    def _get_obs(self) -> np.array:
        """
        get the current state of the environment (positions of nemo)
        :return: dict of object name: coordinates
        """
        return self._nemo.pos()

    def set_render_mode(self, render_mode):
        self.render_mode = render_mode

    def set_movement_penalty(self, pen):
        self._rewards['move'] = -pen

    def reset(self, **kwargs) -> tuple:
        """
        Reset the environment to an initial state.
        """
        if not 'pos_only' in kwargs:
            self.policy = Policy.get_random_policy(self.size)
        self._nemo.set_pos(np.array((1,7), dtype=np.int32))
        return self._get_obs(), {}

    def reset_random(self, **kwargs) -> tuple:
        """
        Reset the environment to an initial random state.
        """
        for _ in range(1000):
            random_position = np.random.randint(0, self.size, size=2)
            if self.is_valid_pos(tuple(random_position)):  
                self._nemo.set_pos(np.array(random_position, dtype=np.int32))
                self.policy = Policy.get_random_policy(self.size)
                return self._get_obs(), {}
        raise ValueError("Failed to find a valid random starting position")
            
    def step(self, action=None) -> tuple:
        """
        Updates the state based on the action.
        return: observation, reward, done (episode), info (for debugging)
        """
        state = self._get_obs()
        if action is None:
            action = self.policy.sample(tuple(state))

        self._nemo.set_pos(self.get_new_state(action=action))

        reward = self.reward()
        done = self.is_terminal()

        return self._get_obs(), reward, done, False, {} # new state after taking an action

    def get_new_state(self, *, state=None, action=None) -> np.array:
        """
        Computes the new state without actually going to that state.
        :param state: current state
        :param action: action to take
        :return: new state
        """
        old_pos = state if state is not None else self._nemo.pos()
        direction = self._action_to_direction[action]
        new_pos = self._nemo.move(old_pos, direction)

        # prevent movement if the new position is in an impassable position
        if self.is_valid_pos(new_pos):
            return new_pos
        return old_pos


    def is_valid_pos(self, state):
        """
        check if the state is in a valid position
        :param state: state
        :return: true if state is not on seaweed, false if it is on seaweed
        """
        if any(np.array_equal(state, v) for k, v in self._objects.items() if k.type() == 'seaweed'):
            return False
        return True

    def get_initial_values(self):
        """
        get an array of state values all set to 0
        :return: 8x8 array of 0s
        """
        return np.zeros([self.size, self.size])

    def reward(self, state=None) -> int:
        """
        Determines the reward for the given state based on Nemos position.
        :param state:
        :return:
        """
        state = np.array(state) if state is not None else self._nemo.pos()
        reward = self._rewards['move']
        for o, pos in self._objects.items():
            if o.type() == 'shark':
                if np.array_equal(state, pos) or np.array_equal(state + np.array([-1, 0]), pos): 
                    reward += self._rewards['shark']
            elif np.array_equal(state, pos):
                reward += self._rewards[o.type()]
        return reward

    def is_terminal(self, state=None) -> bool:
        """
        Checks if the given state is terminal.
        """
        state = state if state is not None else self._nemo.pos()
        return np.array_equal(state, self._objects[self._anemone])

    def get_terminal_state(self) -> np.array:
        """
        :return: the terminal state
        """
        return np.array((7,7), dtype=np.int32)

    def get_non_terminal_states(self):
        """
        :return: the non-terminal states
        """
        raise NotImplementedError

    def render(self) -> None:
        if self.render_mode == "human":
            # initialize if first time rendering
            if not self.window:
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                self.clock = pygame.time.Clock()

            # draw background
            canvas = pygame.image.load('sea.jpg')
            for line in range(1, self.size):
                pygame.draw.line(
                    canvas, (200, 200, 255),
                    (self.square_pix_size * line, 0), (self.square_pix_size * line, self.window_size)
                )
                pygame.draw.line(
                    canvas, (200, 200, 255),
                    (0, self.square_pix_size * line), (self.window_size, self.square_pix_size * line)
                )
            self.window.blit(canvas, canvas.get_rect())

            # draw objects
            for o, pos in self._objects.items():
                GameObject.draw(o, pos, self.window)
            Agent.draw(self._nemo, self._nemo.pos(), self.window)

            # update clock and display
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])

        elif self.render_mode == "text":
            # print grid in text form
            grid = np.full((self.size, self.size), " ")
            for o, pos in self._objects.items():
                grid[pos[1]][pos[0]] = o
            for row in grid:
                print(row)
            print("="*(self.size*4+1))

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def was_closed() -> bool:
        # if user closes pygame window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
    