# Code for random and greedy policies for comparison with mcts algorithm.
import random
from agent import AbstractAgent
from env import GridWorld


class RandomAgent(AbstractAgent):
    def select_action(self, env):
        """
        A policy that selects actions uniformly at random.
        """
        return random.choice(env.action_space)
        

class GreedyAgent(AbstractAgent):
    def select_action(self, env):
        """
        A simple greedy policy that moves the agent closer to the goal.
        Assumes the environment has 'agent_pos' and 'goal_pos' attributes.
        """
        agent_pos, goal_pos, _ = env.get_state()
        best_action = None
        min_distance = float('inf')
        
        for action in env.action_space:
            x, y = agent_pos
            if action == 'up':
                new_pos = (x, y + 1)
            elif action == 'down':
                new_pos = (x, y - 1)
            elif action == 'left':
                new_pos = (x - 1, y)
            elif action == 'right':
                new_pos = (x + 1, y)
            else:
                continue
            
            dist = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])
            if dist < min_distance:
                min_distance = dist
                best_action = action
        return best_action