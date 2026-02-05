from env import GridWorld


class AbstractAgent:

    def __init__(self):
        pass

    def select_action(self, state:GridWorld):
        raise NotImplementedError("This method should be overridden by subclasses")