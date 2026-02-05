# Main Monte Carlo code and sim. This chooses actions and updates statistics to be plugged into the environment based on "current" states.

import math


class Node:
    """A node in the Monte Carlo Tree Search."""

    def __init__(self, state, parent=None):
        """Initialize the node with state and parent."""
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = state.actions()

    @property
    def q_value(self) -> float:
        """Average reward of the node."""
        return self.total_reward / self.visits if self.visits > 0 else 0.0


def uct_value(parent: Node, child: Node, exploration_param: float = math.sqrt(2)) -> float:
    """Calculate the UCT value for a node."""
    return child.q_value + exploration_param * math.sqrt(
        math.log(parent.visits) / (child.visits)
    )  # Q + C * sqrt(ln(N) / n)


def select(node: Node) -> Node:
    """Select the child node with the highest UCT value."""
    while not node.state.is_terminal():
        if node.untried_actions:
            return node
        node = max(node.children, key=lambda n: uct_value(node, n))

    return node


def expand(node: Node) -> Node:
    """Expansion: Add a new child node for an untried action."""
    action = node.untried_actions.pop()
    next_state = node.state.take_action(action)
    child_node = Node(state=next_state, parent=node)
    node.children.append(child_node)
    return child_node


def simulate(state) -> float:
    """Roll out with default policy."""
    while not state.is_terminal():
        pass  # TODO:
    return state.reward()


def backpropogate():
    pass  # TODO


def mcts():
    pass  # TODO

