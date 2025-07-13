from abc import ABC, abstractmethod
from typing import Tuple

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.transform import translate_longitudinally_and_laterally
from shapely import Polygon


class AbstractEnvironmentArea(ABC):
    """
    Abstract class for defining an environment area in a Gym simulation.
    The area defines the area used for the observation, reward, and other simulation components.
    """

    @abstractmethod
    def get_global_origin(self, ego_pose: StateSE2) -> StateSE2:
        """
        Given the global ego pose, returns the global origin of the environment area.
        :param ego_pose: Global ego pose in the environment.
        :return: Global origin of the environment area.
        """
        pass

    @abstractmethod
    def get_global_polygon(self, ego_pose: StateSE2) -> Polygon:
        """
        Given the global ego pose, returns the environment area as 2D polygon.
        :param ego_pose: Global ego pose in the environment.
        :return: 2D polygon representing the environment area.
        """
        pass


class RectangleEnvironmentArea(AbstractEnvironmentArea):
    def __init__(
        self,
        front: float = 78.0,
        back: float = 50.0,
        left: float = 64.0,
        right: float = 64.0,
    ) -> None:
        """
        Initializes a rectangular environment area.
        :param front: extent of area in the front of the ego vehicle [m], defaults to 78.0
        :param back: extent of area in back of the ego vehicle [m], defaults to 50.0
        :param left: extent of area to the left of the ego vehicle [m], defaults to 64.0
        :param right: extent of area to the right of the ego vehicle [m], defaults to 64.0
        """
        self._front = front
        self._back = back
        self._left = left
        self._right = right

    @property
    def frame(self) -> Tuple[float, float]:
        """
        Returns the dimensions of the rectangle as a tuple (width, height).
        :return: Tuple of width and height of the rectangle.
        """
        return (self._left + self._right), (self._front + self._back)

    def get_global_origin(self, ego_pose: StateSE2) -> StateSE2:
        """Inherited, see superclass."""
        width, height = self.frame
        longitudinal_offset = (height / 2.0) - self._back
        lateral_offset = (width / 2.0) - self._right
        return translate_longitudinally_and_laterally(ego_pose, longitudinal_offset, lateral_offset)

    def get_global_polygon(self, ego_pose: StateSE2) -> Polygon:
        """Inherited, see superclass."""
        return Polygon(
            [
                tuple(translate_longitudinally_and_laterally(ego_pose, self._front, self._left).point),  # front left
                tuple(translate_longitudinally_and_laterally(ego_pose, self._front, -self._right).point),  # front right
                tuple(translate_longitudinally_and_laterally(ego_pose, -self._back, -self._right).point),  # rear right
                tuple(translate_longitudinally_and_laterally(ego_pose, -self._back, self._left).point),  # rear left
            ]
        )
