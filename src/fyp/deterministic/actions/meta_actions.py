class MetaAction():

    def __init__(self, action, action_sequence):
        self.action = action
        self.action_sequence = action_sequence

    def __eq__(self, other):
        return self.action == other.action and self.action_sequence == other.action_sequence

    def __key(self):
        return (self.action, *self.action_sequence)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return f"Meta Action: replace {self.action} by {self.action_sequence}"

    # def __repr__(self):
    #     return str(self)