# size=10, slip_prob=0.1, num_obstacles=5 Implementation of the grid world, including obstacles and goal and stochastic movement and dynamics pulled from config.py

import random

from config import Config


class GridWorld:
    def __init__(self, config: Config):
        self.size = config.grid_size
        self.slip_prob = config.slip_prob
        self.num_obstacles = config.num_obstacles
        self.movement_reward = config.movement_reward
        self.obstacle_penalty = config.obstacle_penalty
        self.goal_reward = config.goal_reward
        self.obstacle_move_prob = config.obstacle_move_prob
        self.goal_move_prob = config.goal_move_prob
        self.config = config
        self.action_space = ["u", "d", "l", "r"]
        self.state_history = []  # To keep track of states for visualization
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]  # START AT TOP-LEFT CORNER - COULD BE MODIFIED TO RANDOM START
        self.goal_pos = [self.size - 1, self.size - 1]  # Goal at bottom-right corner - COULD BE MODIFIED TO RANDOM GOAL
        self.obstacles = self.generate_obstacles()
        self.state_history = [self.get_state()]  # Clear state history on reset and add initial state
        return self.agent_pos

    def generate_obstacles(self):
        obs = set()
        while len(obs) < self.num_obstacles:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if pos != tuple(self.agent_pos) and pos != tuple(self.goal_pos) and pos not in obs:
                obs.add(pos)
        return obs

    def get_state(self):
        return (tuple(self.agent_pos), tuple(self.goal_pos), tuple(map(tuple, self.obstacles)))

    def save_state_to_history(self):
        self.state_history.append(self.get_state())

    def move_goal(self):
        if random.random() < self.goal_move_prob:
            direction = random.choice(self.action_space)
            x, y = self.goal_pos
            if direction == "u" and y > 0 and [x, y - 1] != self.agent_pos and [x, y - 1] not in self.obstacles:
                self.goal_pos[1] -= 1
            elif direction == "d" and y < self.size - 1 and [x, y + 1] != self.agent_pos and [x, y + 1] not in self.obstacles:
                self.goal_pos[1] += 1
            elif direction == "l" and x > 0 and [x - 1, y] != self.agent_pos and [x - 1, y] not in self.obstacles:
                self.goal_pos[0] -= 1
            elif direction == "r" and x < self.size - 1 and [x + 1, y] != self.agent_pos and [x + 1, y] not in self.obstacles:
                self.goal_pos[0] += 1

    def step(self, action):
        # agent_position, goal_position, obstacles = self.get_state()
        # self.agent_pos = list(agent_position)
        self.obstacles = [list(obs) for obs in self.obstacles]

        if random.random() < self.slip_prob:
            action = random.choice(self.action_space)

        self.move_agent(action)
        self.move_obstacles()
        self.move_goal()

        reward = self.movement_reward

        done = False
        if self.agent_pos in self.obstacles:
            reward = self.obstacle_penalty
            done = True
        elif self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            done = True

        self.save_state_to_history()
        return self.get_state(), reward, done

    def move_agent(self, action):
        if action == "u" and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == "d" and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        elif action == "l" and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == "r" and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
        else:
            pass  # Invalid move, agent stays in place

    def move_obstacles(self):
        for obs in self.obstacles:
            if random.random() < self.obstacle_move_prob:
                direction = random.choice(self.action_space)
                if (
                    direction == "u"
                    and obs[1] > 0
                    and [obs[0], obs[1] - 1] != self.agent_pos
                    and [obs[0], obs[1] - 1] != self.goal_pos
                ):
                    obs[1] -= 1
                elif (
                    direction == "d"
                    and obs[1] < self.size - 1
                    and [obs[0], obs[1] + 1] != self.agent_pos
                    and [obs[0], obs[1] + 1] != self.goal_pos
                ):
                    obs[1] += 1
                elif (
                    direction == "l"
                    and obs[0] > 0
                    and [obs[0] - 1, obs[1]] != self.agent_pos
                    and [obs[0] - 1, obs[1]] != self.goal_pos
                ):
                    obs[0] -= 1
                elif (
                    direction == "r"
                    and obs[0] < self.size - 1
                    and [obs[0] + 1, obs[1]] != self.agent_pos
                    and [obs[0] + 1, obs[1]] != self.goal_pos
                ):
                    obs[0] += 1

    def clone(self):
        """Create a deep copy of the environment for simulation purposes."""
        clone_env = GridWorld(config=self.config)
        clone_env.agent_pos = self.agent_pos.copy()
        clone_env.goal_pos = self.goal_pos.copy()
        clone_env.obstacles = [list(obs) for obs in self.obstacles]
        clone_env.state_history = []  # Don't copy history for simulations
        return clone_env
