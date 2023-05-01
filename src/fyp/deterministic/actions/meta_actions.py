from enum import Enum

class BaseMetaActions(Enum):
    ADD_TRANSITION = 0
    REMOVE_TRANSITION = 1
    INCREASE_REWARD = 2
    DECREASE_REWARD = 3


class MetaAction():
    pass

class MetaActionT(MetaAction):

    def __init__(self, action, action_sequence):
        self.action = action
        if action_sequence is None:
            self.action_sequence = [action]
        else:
            self.action_sequence = action_sequence

    def __eq__(self, other):
        if not hasattr(other, "action"):
            return False
        return self.action == other.action and self.action_sequence == other.action_sequence

    def __key(self):
        if self.action_sequence is None:
            return (self.action, self.action)
        return (self.action, *self.action_sequence)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return f"Meta Action: replace {self.action} by {self.action_sequence}"

class MetaActionR(MetaAction):

    def __init__(self, reward):
        self.reward = reward

    def __eq__(self, other):
        if not hasattr(other, "reward"):
            return False
        return self.reward == other.reward
    
    def __key(self):
        return self.reward
    
    def __hash__(self):
        return hash(self.__key())
    
    def __str__(self):
        return f"Meta Action: increase reward to {self.reward}"