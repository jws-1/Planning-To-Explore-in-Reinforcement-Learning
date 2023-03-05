from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class MetaAction(Enum):
    PLACE_OBJECT = 0
    REMOVE_OBJECT = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3
    INCREASE_TRANSITION_PROBABILITY = 4
    DECREASE_TRANSITION_PROBABILITY = 5

ACTION_MODIFIERS = {
    Action.UP: (0,1),
    Action.DOWN: (0, -1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
    MetaAction.PLACE_OBJECT: (0, 0),
    MetaAction.REMOVE_OBJECT: (0, 0),
    MetaAction.INCREASE_REWARD: (0,0),
    MetaAction.DECREASE_REWARD: (0,0)
}
