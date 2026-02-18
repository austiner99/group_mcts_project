# Main Monte Carlo code and sim. This chooses actions and updates statistics to be plugged into the environment based on "current" states.

import math
import random

from agent import AbstractAgent
from config import Config


class Node:
    """A node in the Monte Carlo Tree Search."""

    def __init__(self, env, policy="", parent=None, action=None):
        """Initialize the node with state and parent."""
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.policy = policy
        self.untried_actions = list(env.action_space)

    @property
    def q_value(self) -> float:
        """Average reward of the node."""
        return self.total_reward / self.visits if self.visits > 0 else -float("inf")


def is_terminal_env(env):
    """Check if the env is in a terminal state."""
    agent_pos, goal_pos, obstacles = env.get_state()
    return agent_pos == goal_pos or agent_pos in obstacles


def uct_value(parent: Node, child: Node, exploration_param: float = math.sqrt(2)) -> float:
    """Calculate the UCT value for a node."""
    return child.q_value + exploration_param * math.sqrt(
        math.log(parent.visits) / (child.visits)
    )  # Q + C * sqrt(ln(N) / n)


def select(node: Node) -> Node:
    """Select the child node with the highest UCT value."""
    while not is_terminal_env(node.env):
        if node.untried_actions:
            return node
        node = max(node.children, key=lambda n: uct_value(node, n))

    return node


def expand(node: Node, env) -> Node:
    """Expansion: Add a new child node for an untried action."""
    action = node.untried_actions.pop()
    pi = node.parent.policy + action if node.parent else action
    child_node = Node(env=env, policy=pi, parent=node, action=action)
    node.children.append(child_node)
    return child_node


def rollout_policy(env):
    """Random rollout policy."""
    return random.choice(env.action_space)


def simulate(env, first_action, rollout_depth: int = 50) -> float:
    """Roll out with default policy."""
    total_reward = 0.0
    depth = 0

    action = rollout_policy(env)
    _, reward, done = env.step(first_action)
    total_reward += reward
    depth += 1

    while not is_terminal_env(env) and depth < rollout_depth:
        action = rollout_policy(env)
        _, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
        depth += 1
    return total_reward


def backpropogate(node: Node, reward: float) -> None:
    """Backpropagate the reward up the tree."""
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


def mcts(
    root_env,
    iterations: int = 500,
    rollout_depth: int = 50,
):
    """Perform Monte Carlo Tree Search and return the best action."""
    root = Node(root_env.clone())
    actions = []
    for _ in root_env.action_space:
        actions.append(expand(root, root_env))

    for _ in range(iterations):
        node = random.choice(actions)  # Randomly select one of the expanded nodes
        first_action = node.action

        reward = simulate(root_env.clone(), first_action, rollout_depth=rollout_depth)
        backpropogate(node, reward)

    if not root.children:
        return random.choice(root_env.action_space)  # No children, choose random action

    best_child = max(root.children, key=lambda n: n.q_value)  # Choose child with highest average reward
    return best_child.action


class MCTSRandomAgent(AbstractAgent):
    """Monte Carlo Tree Search agent."""

    def __init__(
        self,
        config: Config,
    ):
        """Initialize the MCTS agent with parameters."""
        self.iterations = config.mcts_iterations
        self.rollout_depth = config.mcts_rollout_depth

    def select_action(self, env):
        """Select an action using Monte Carlo Tree Search."""
        return mcts(
            env,
            iterations=self.iterations,
            rollout_depth=self.rollout_depth,
        )
