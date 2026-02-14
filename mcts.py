# Main Monte Carlo code and sim. This chooses actions and updates statistics to be plugged into the environment based on "current" states.

import math
import random

from agent import AbstractAgent


class Node:
    """A node in the Monte Carlo Tree Search."""

    def __init__(self, env, parent=None, action=None):
        """Initialize the node with state and parent."""
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = list(env.action_space)

    @property
    def q_value(self) -> float:
        """Average reward of the node."""
        return self.total_reward / self.visits if self.visits > 0 else float("-inf")


def is_terminal_env(env):
    """Check if the env is in a terminal state."""
    agent_pos, goal_pos, obstacles = env.get_state()
    return agent_pos == goal_pos or agent_pos in obstacles


def uct_value(parent: Node, child: Node, exploration_param: float = math.sqrt(2)) -> float:
    """Calculate the UCT value for a node."""
    if child.visits == 0:
        return float("inf")  # Prioritize unvisited nodes
    return child.q_value + exploration_param * math.sqrt(
        math.log(parent.visits) / (child.visits)
    )  # Q + C * sqrt(ln(N) / n)


def select(node: Node, exploration_param: float = math.sqrt(2)) -> Node:
    """Select the child node with the highest UCT value."""
    while not is_terminal_env(node.env):
        if node.untried_actions:
            return node
        node = max(node.children, key=lambda n: uct_value(node, n, exploration_param=exploration_param))

    return node


def expand(node: Node) -> Node:
    """Expansion: Add a new child node for an untried action."""
    action = node.untried_actions.pop()
    next_env = node.env.clone() # Another issue might be here. This is saving the stochasticity of the environment which block it from exploring different states
    next_env.step(action)
    child_node = Node(env=next_env, parent=node, action=action)
    node.children.append(child_node)
    return child_node


def rollout_policy(env, epsilon=0.1):
    """Epsilon-greedy rollout policy."""
    if epsilon >= 1.0 or random.random() < epsilon:
        return random.choice(env.action_space)  # Explore
    # Exploit: Choose action that moves towards the goal
    agent_pos, goal_pos, obstacles = env.get_state()
    best_action = None
    best_distance = float("inf")
    for action in env.action_space:
        x, y = agent_pos
        if action == "u":
            y -= 1
        elif action == "d":
            y += 1
        elif action == "l":
            x -= 1
        elif action == "r":
            x += 1
        else:
            continue
        dist = abs(x - goal_pos[0]) + abs(y - goal_pos[1])
        if dist < best_distance and (x, y) not in obstacles:
            best_distance = dist
            best_action = action
    return best_action   # If all actions lead to obstacles, return None

def simulate(env, rollout_depth: int = 50, epsilon=0.1) -> float:
    """Roll out with default policy."""
    total_reward = 0.0
    depth = 0
    while not is_terminal_env(env) and depth < rollout_depth:
        action = rollout_policy(env, epsilon=epsilon)
        _, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
        depth += 1
    return total_reward

# I *think* this is where an issue is. Nodes aren't being created when you do rollout, so the backprop is only updating the starting node and its parents. 
def backpropogate(node: Node, reward: float) -> None:
    """Backpropagate the reward up the tree."""
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent


def mcts(
    root_env,
    iterations: int = 500,
    exploration_param: float = math.sqrt(2),
    rollout_depth: int = 50,
    epsilon: float = 0.1,
):
    """Perform Monte Carlo Tree Search and return the best action."""
    root = Node(root_env.clone())
    for _ in range(iterations):
        node = select(root, exploration_param=exploration_param)
        if not is_terminal_env(node.env) and node.untried_actions:
            node = expand(node)
        reward = simulate(node.env.clone(), rollout_depth=rollout_depth, epsilon=epsilon)
        backpropogate(node, reward)

    if not root.children:
        return random.choice(root_env.action_space)  # No children, choose random action

    best_child = max(root.children, key=lambda n: n.q_value)  # Choose child with highest average reward
    return best_child.action


class MCTSAgent(AbstractAgent):
    """Monte Carlo Tree Search agent."""

    def __init__(
        self, iterations: int = 500, exploration_param: float = 1.4, rollout_depth: int = 10, epsilon: float = 0.1
    ):
        """Initialize the MCTS agent with parameters."""
        self.iterations = iterations
        self.exploration_param = exploration_param
        self.rollout_depth = rollout_depth
        self.epsilon = epsilon

    def select_action(self, env):
        """Select an action using Monte Carlo Tree Search."""
        return mcts(
            env,
            iterations=self.iterations,
            exploration_param=self.exploration_param,
            rollout_depth=self.rollout_depth,
            epsilon=self.epsilon,
        )
