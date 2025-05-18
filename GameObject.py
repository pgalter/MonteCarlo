import pygame
import numpy as np

class GameObject:
    def __init__(self, type: str, filename: str, grid_square_size: float,
                 max_pos: int=None, double:bool = False):
        """
        Get a GameObject, with a coordinate position and visual representation.
        :param filename: name of the .png file that contains the sprite image
        :param grid_square_size: size of one grid square in pixels
        :param max_pos: maximum possible coordinates the object can exist on (6,6 for example)
        :param double: whether the object covers two positions (e.g., sharks)
        """
        self._type = type
        self._max_pos = max_pos if max_pos else None
        self._maxsize = grid_square_size

        # loading and scaling the image
        img = pygame.image.load(filename)
        transform_factor = grid_square_size / max(img.get_width(), img.get_height())
        if double: transform_factor *= 2
        self._sprite = pygame.transform.scale(
            img,
            (img.get_width() * transform_factor, img.get_height() * transform_factor)
        )

    def type(self):
        """
        :return: type of the object
        """
        return self._type

    def _pix_pos(self, pos) -> np.array:
        """
        get the position of top left point of object in pixels
        :return: object's position as vector of pixel coordinates
        """
        return (self._maxsize * pos
                    + (0.5 * np.array(
                        [self._maxsize - self._sprite.get_height(), self._maxsize - self._sprite.get_height()]
                    ))
                )

    def draw(self, pos, screen: pygame.Surface):
        """
        draw object on screen at its current position
        :param screen: pygame surface to draw on
        """
        screen.blit(self._sprite, self._pix_pos(pos))

class Agent(GameObject):
    def __init__(self, type: str, filename: str, grid_square_size: float, max_pos: int = None, double: bool = False):
        super().__init__(type, filename, grid_square_size, max_pos, double)
        self._pos = None

    def pos(self):
        return self._pos

    def set_pos(self, pos):
        self._pos = pos

    def move(self, pos, move: np.array) -> np.array:
        """
        update position of object by adding "move" to its position
        :param pos: current position
        :param move: move vector
        :return: new position
        """
        return np.clip(pos + move, 0, self._max_pos - 1, dtype=np.int32)
        # clip to make sure the agent stays inside the grid